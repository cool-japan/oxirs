//! # OxiRS Rule Engine - Getting Started Guide
//!
//! This example provides a gentle introduction to the OxiRS Rule Engine.
//! Perfect for newcomers who want to understand the basics.
//!
//! Run with: cargo run --example getting_started

use anyhow::Result;
use oxirs_rule::*;

fn main() -> Result<()> {
    println!("🌟 Welcome to OxiRS Rule Engine!");
    println!("=================================\n");

    // Step 1: Create a rule engine
    step1_create_engine()?;
    
    // Step 2: Add simple rules
    step2_add_rules()?;
    
    // Step 3: Add facts and run inference
    step3_inference()?;
    
    // Step 4: Query the knowledge base
    step4_querying()?;

    println!("🎉 Congratulations! You've completed the getting started guide.");
    println!("Next steps:");
    println!("  - Try the comprehensive_tutorial example");
    println!("  - Explore RDFS, OWL, and SWRL reasoning");
    println!("  - Use debugging tools for complex scenarios");
    
    Ok(())
}

fn step1_create_engine() -> Result<()> {
    println!("📖 Step 1: Creating a Rule Engine");
    println!("--------------------------------");
    
    // Create a new rule engine instance
    let mut engine = RuleEngine::new();
    
    println!("✅ Created a new rule engine!");
    println!("   The engine supports:");
    println!("   - Forward chaining (derive new facts from rules)");
    println!("   - Backward chaining (prove goals by finding supporting facts)");
    println!("   - RETE network (efficient pattern matching)");
    println!();
    
    Ok(())
}

fn step2_add_rules() -> Result<()> {
    println!("📖 Step 2: Adding Rules");
    println!("----------------------");
    
    let mut engine = RuleEngine::new();
    
    // Create your first rule: "If X is a cat, then X is an animal"
    let cat_rule = Rule {
        name: "cats_are_animals".to_string(),
        body: vec![
            // Rule body: cat(X)
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("cat".to_string()),
            }
        ],
        head: vec![
            // Rule head: animal(X)
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("animal".to_string()),
            }
        ],
    };
    
    // Add the rule to the engine
    engine.add_rule(cat_rule);
    
    println!("✅ Added rule: cats_are_animals");
    println!("   If ?X type cat → ?X type animal");
    
    // Add another rule: "If X is an animal, then X is alive"
    let animal_rule = Rule {
        name: "animals_are_alive".to_string(),
        body: vec![
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("animal".to_string()),
            }
        ],
        head: vec![
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("property".to_string()),
                object: Term::Constant("alive".to_string()),
            }
        ],
    };
    
    engine.add_rule(animal_rule);
    
    println!("✅ Added rule: animals_are_alive");
    println!("   If ?X type animal → ?X property alive");
    println!();
    
    Ok(())
}

fn step3_inference() -> Result<()> {
    println!("📖 Step 3: Adding Facts and Running Inference");
    println!("---------------------------------------------");
    
    let mut engine = RuleEngine::new();
    
    // Add the rules from step 2
    engine.add_rule(Rule {
        name: "cats_are_animals".to_string(),
        body: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("cat".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("animal".to_string()),
        }],
    });
    
    engine.add_rule(Rule {
        name: "animals_are_alive".to_string(),
        body: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("animal".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("property".to_string()),
            object: Term::Constant("alive".to_string()),
        }],
    });
    
    // Add some facts about specific cats
    let facts = vec![
        RuleAtom::Triple {
            subject: Term::Constant("whiskers".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("cat".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("fluffy".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("cat".to_string()),
        },
    ];
    
    println!("📝 Initial facts:");
    for fact in &facts {
        if let RuleAtom::Triple { subject, predicate, object } = fact {
            if let (Term::Constant(s), Term::Constant(p), Term::Constant(o)) = (subject, predicate, object) {
                println!("   {} {} {}", s, p, o);
            }
        }
    }
    
    // Run forward chaining to derive new facts
    println!("\n🔄 Running forward chaining...");
    let results = engine.forward_chain(&facts)?;
    
    println!("\n🧠 Derived facts:");
    for result in &results {
        if let RuleAtom::Triple { subject, predicate, object } = result {
            if let (Term::Constant(s), Term::Constant(p), Term::Constant(o)) = (subject, predicate, object) {
                // Only show derived facts (not the original ones)
                if !facts.contains(result) {
                    println!("   {} {} {} (derived)", s, p, o);
                }
            }
        }
    }
    
    println!("\n💡 What happened?");
    println!("   1. We started with: whiskers type cat, fluffy type cat");
    println!("   2. Rule 1 derived: whiskers type animal, fluffy type animal");
    println!("   3. Rule 2 derived: whiskers property alive, fluffy property alive");
    println!();
    
    Ok(())
}

fn step4_querying() -> Result<()> {
    println!("📖 Step 4: Querying the Knowledge Base");
    println!("-------------------------------------");
    
    let mut engine = RuleEngine::new();
    
    // Set up the same rules and facts as before
    engine.add_rule(Rule {
        name: "cats_are_animals".to_string(),
        body: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("cat".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("animal".to_string()),
        }],
    });
    
    engine.add_rule(Rule {
        name: "animals_are_alive".to_string(),
        body: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("animal".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("property".to_string()),
            object: Term::Constant("alive".to_string()),
        }],
    });
    
    let facts = vec![
        RuleAtom::Triple {
            subject: Term::Constant("whiskers".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("cat".to_string()),
        },
    ];
    
    // Add facts to the engine
    engine.add_facts(facts);
    
    // Use backward chaining to answer specific questions
    println!("🔍 Asking questions using backward chaining:");
    
    // Question 1: Is whiskers an animal?
    let query1 = RuleAtom::Triple {
        subject: Term::Constant("whiskers".to_string()),
        predicate: Term::Constant("type".to_string()),
        object: Term::Constant("animal".to_string()),
    };
    
    let answer1 = engine.backward_chain(&query1)?;
    println!("   Q: Is whiskers an animal?");
    println!("   A: {} ✅", if answer1 { "Yes" } else { "No" });
    
    // Question 2: Is whiskers alive?
    let query2 = RuleAtom::Triple {
        subject: Term::Constant("whiskers".to_string()),
        predicate: Term::Constant("property".to_string()),
        object: Term::Constant("alive".to_string()),
    };
    
    let answer2 = engine.backward_chain(&query2)?;
    println!("   Q: Is whiskers alive?");
    println!("   A: {} ✅", if answer2 { "Yes" } else { "No" });
    
    // Question 3: Is whiskers a dog? (should be false)
    let query3 = RuleAtom::Triple {
        subject: Term::Constant("whiskers".to_string()),
        predicate: Term::Constant("type".to_string()),
        object: Term::Constant("dog".to_string()),
    };
    
    let answer3 = engine.backward_chain(&query3)?;
    println!("   Q: Is whiskers a dog?");
    println!("   A: {} ❌", if answer3 { "Yes" } else { "No" });
    
    println!("\n💡 How backward chaining works:");
    println!("   - To prove 'whiskers is alive', the engine:");
    println!("   - Looks for rules that conclude 'X property alive'");
    println!("   - Finds the rule: 'if X type animal → X property alive'");
    println!("   - Now needs to prove 'whiskers type animal'");
    println!("   - Finds the rule: 'if X type cat → X type animal'");
    println!("   - Checks if 'whiskers type cat' is a known fact");
    println!("   - Since it is, the proof succeeds!");
    println!();
    
    Ok(())
}

/// Bonus: Common patterns and best practices
#[allow(dead_code)]
fn bonus_patterns() -> Result<()> {
    println!("🎯 Bonus: Common Patterns and Best Practices");
    println!("===========================================");
    
    let mut engine = RuleEngine::new();
    
    // Pattern 1: Conditional rules with multiple conditions
    let conditional_rule = Rule {
        name: "smart_adult_rule".to_string(),
        body: vec![
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("person".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("age".to_string()),
                object: Term::Variable("A".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("iq".to_string()),
                object: Term::Variable("I".to_string()),
            },
            // Note: In practice, you'd use SWRL built-ins for comparisons
        ],
        head: vec![
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("category".to_string()),
                object: Term::Constant("qualified_candidate".to_string()),
            },
        ],
    };
    
    engine.add_rule(conditional_rule);
    
    // Pattern 2: Transitive rules (A -> B, B -> C implies A -> C)
    let transitive_rule = Rule {
        name: "transitivity".to_string(),
        body: vec![
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("manages".to_string()),
                object: Term::Variable("Y".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Variable("Y".to_string()),
                predicate: Term::Constant("manages".to_string()),
                object: Term::Variable("Z".to_string()),
            },
        ],
        head: vec![
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("indirectly_manages".to_string()),
                object: Term::Variable("Z".to_string()),
            },
        ],
    };
    
    engine.add_rule(transitive_rule);
    
    println!("✅ Added common rule patterns:");
    println!("   - Conditional rules (multiple body conditions)");
    println!("   - Transitive rules (chain relationships)");
    
    // Best practice: Organize rules by domain
    println!("\n📚 Best Practices:");
    println!("   1. Use descriptive rule names");
    println!("   2. Group related rules together");
    println!("   3. Test with small datasets first");
    println!("   4. Use debugging tools for complex scenarios");
    println!("   5. Consider performance for large datasets");
    println!("   6. Validate rule logic before deployment");
    
    Ok(())
}