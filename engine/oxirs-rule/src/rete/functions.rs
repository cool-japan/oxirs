//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use once_cell::sync::Lazy;
use scirs2_core::metrics::{Counter, Gauge};

pub static TOKEN_CLONES: Lazy<Counter> =
    Lazy::new(|| Counter::new("rete_token_clones".to_string()));
pub static ACTIVE_TOKENS: Lazy<Gauge> = Lazy::new(|| Gauge::new("rete_active_tokens".to_string()));
/// Unique identifier for RETE nodes
pub type NodeId = usize;
#[cfg(test)]
mod tests {
    use crate::{
        rete::ReteNetwork,
        rete_enhanced::{ConflictResolution, MemoryStrategy},
        Rule, RuleAtom, Term,
    };
    #[test]
    fn test_rete_network_creation() {
        let network = ReteNetwork::new();
        let stats = network.get_stats();
        assert_eq!(stats.total_nodes, 1);
        assert_eq!(stats.alpha_nodes, 0);
        assert_eq!(stats.beta_nodes, 0);
        assert_eq!(stats.production_nodes, 0);
    }
    #[test]
    fn test_simple_rule_compilation() -> Result<(), Box<dyn std::error::Error>> {
        let mut network = ReteNetwork::new();
        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("human".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("mortal".to_string()),
            }],
        };
        network.add_rule(&rule)?;
        let stats = network.get_stats();
        assert!(stats.alpha_nodes > 0);
        assert!(stats.production_nodes > 0);
        Ok(())
    }
    #[test]
    fn test_fact_processing() -> Result<(), Box<dyn std::error::Error>> {
        let mut network = ReteNetwork::new();
        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("human".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("mortal".to_string()),
            }],
        };
        network.add_rule(&rule)?;
        let fact = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        };
        let derived = network.add_fact(fact)?;
        assert!(!derived.is_empty());
        let expected = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("mortal".to_string()),
        };
        assert!(derived.contains(&expected));
        Ok(())
    }
    #[test]
    fn test_forward_chaining() -> Result<(), Box<dyn std::error::Error>> {
        let mut network = ReteNetwork::new();
        let rule = Rule {
            name: "simple_rule".to_string(),
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
        network.add_rule(&rule)?;
        let initial_facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Constant("mary".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("mary".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Constant("bob".to_string()),
            },
        ];
        let all_facts = network.forward_chain(initial_facts)?;
        let expected1 = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("ancestor".to_string()),
            object: Term::Constant("mary".to_string()),
        };
        let expected2 = RuleAtom::Triple {
            subject: Term::Constant("mary".to_string()),
            predicate: Term::Constant("ancestor".to_string()),
            object: Term::Constant("bob".to_string()),
        };
        assert!(all_facts.contains(&expected1));
        assert!(all_facts.contains(&expected2));
        Ok(())
    }
    #[test]
    fn test_enhanced_beta_join() -> Result<(), Box<dyn std::error::Error>> {
        let mut network = ReteNetwork::with_strategies(
            MemoryStrategy::LimitCount(100),
            ConflictResolution::Specificity,
        );
        network.debug_mode = true;
        let rule = Rule {
            name: "parent_grandparent".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("parent".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("parent".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("grandparent".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        };
        println!("Adding rule to network...");
        network.add_rule(&rule)?;
        println!("Rule added successfully");
        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Constant("mary".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("mary".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Constant("alice".to_string()),
            },
        ];
        println!("Input facts: {facts:?}");
        let results = network.forward_chain(facts)?;
        println!("Actual results: {results:?}");
        let expected = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("grandparent".to_string()),
            object: Term::Constant("alice".to_string()),
        };
        println!("Expected result: {expected:?}");
        assert!(results.contains(&expected));
        let stats = network.get_enhanced_stats();
        assert!(stats.total_beta_joins > 0);
        assert!(stats.successful_beta_joins > 0);
        assert!(stats.enhanced_nodes > 0);
        Ok(())
    }
    #[test]
    fn test_memory_management() -> Result<(), Box<dyn std::error::Error>> {
        let mut network = ReteNetwork::with_strategies(
            MemoryStrategy::LimitCount(5),
            ConflictResolution::Recency,
        );
        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("hasProperty".to_string()),
                    object: Term::Variable("P".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("P".to_string()),
                    predicate: Term::Constant("type".to_string()),
                    object: Term::Variable("T".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("hasTypedProperty".to_string()),
                object: Term::Variable("T".to_string()),
            }],
        };
        network.add_rule(&rule)?;
        let mut facts = Vec::new();
        for i in 0..20 {
            facts.push(RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{i}")),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Constant(format!("prop_{i}")),
            });
            facts.push(RuleAtom::Triple {
                subject: Term::Constant(format!("prop_{i}")),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant(format!("type_{val}", val = i % 3)),
            });
        }
        network.forward_chain(facts)?;
        let stats = network.get_enhanced_stats();
        println!(
            "Memory management stats: memory_evictions={}, peak_memory_usage={}, enhanced_nodes={}",
            stats.memory_evictions, stats.peak_memory_usage, stats.enhanced_nodes
        );
        assert!(stats.memory_evictions > 0);
        assert!(stats.enhanced_nodes > 0);
        Ok(())
    }
    #[test]
    fn test_conflict_resolution() -> Result<(), Box<dyn std::error::Error>> {
        let mut network =
            ReteNetwork::with_strategies(MemoryStrategy::Unlimited, ConflictResolution::Priority);
        network.set_debug_mode(true);
        let rule = Rule {
            name: "priority_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };
        network.add_rule(&rule)?;
        let fact = RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("p".to_string()),
            object: Term::Constant("b".to_string()),
        };
        let results = network.add_fact(fact)?;
        assert!(!results.is_empty());
        Ok(())
    }
    #[test]
    fn test_comparison_operator_greater_than() -> Result<(), Box<dyn std::error::Error>> {
        let mut network = ReteNetwork::new();
        network.set_debug_mode(true);
        let rule = Rule {
            name: "isAdult".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("person".to_string()),
                    predicate: Term::Constant(":hasAge".to_string()),
                    object: Term::Variable("age".to_string()),
                },
                RuleAtom::GreaterThan {
                    left: Term::Variable("age".to_string()),
                    right: Term::Literal("18".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("person".to_string()),
                predicate: Term::Constant(":isAdult".to_string()),
                object: Term::Literal("true".to_string()),
            }],
        };
        network.add_rule(&rule)?;
        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant(":hasAge".to_string()),
            object: Term::Literal("20".to_string()),
        }];
        let result = network.forward_chain(facts)?;
        let expected = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant(":isAdult".to_string()),
            object: Term::Literal("true".to_string()),
        };
        assert!(
            result.contains(&expected),
            "Expected {:?} to be in {:?}",
            expected,
            result
        );
        Ok(())
    }
    #[test]
    fn test_comparison_operator_less_than() -> Result<(), Box<dyn std::error::Error>> {
        let mut network = ReteNetwork::new();
        let rule = Rule {
            name: "isMinor".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("person".to_string()),
                    predicate: Term::Constant(":hasAge".to_string()),
                    object: Term::Variable("age".to_string()),
                },
                RuleAtom::LessThan {
                    left: Term::Variable("age".to_string()),
                    right: Term::Literal("18".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("person".to_string()),
                predicate: Term::Constant(":isMinor".to_string()),
                object: Term::Literal("true".to_string()),
            }],
        };
        network.add_rule(&rule)?;
        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant(":hasAge".to_string()),
            object: Term::Literal("15".to_string()),
        }];
        let result = network.forward_chain(facts)?;
        let expected = RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant(":isMinor".to_string()),
            object: Term::Literal("true".to_string()),
        };
        assert!(
            result.contains(&expected),
            "Expected {:?} to be in {:?}",
            expected,
            result
        );
        Ok(())
    }
    #[test]
    fn test_comparison_operator_filter_fails() -> Result<(), Box<dyn std::error::Error>> {
        let mut network = ReteNetwork::new();
        let rule = Rule {
            name: "isAdult".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("person".to_string()),
                    predicate: Term::Constant(":hasAge".to_string()),
                    object: Term::Variable("age".to_string()),
                },
                RuleAtom::GreaterThan {
                    left: Term::Variable("age".to_string()),
                    right: Term::Literal("18".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("person".to_string()),
                predicate: Term::Constant(":isAdult".to_string()),
                object: Term::Literal("true".to_string()),
            }],
        };
        network.add_rule(&rule)?;
        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("bob".to_string()),
            predicate: Term::Constant(":hasAge".to_string()),
            object: Term::Literal("10".to_string()),
        }];
        let result = network.forward_chain(facts)?;
        let not_expected = RuleAtom::Triple {
            subject: Term::Constant("bob".to_string()),
            predicate: Term::Constant(":isAdult".to_string()),
            object: Term::Literal("true".to_string()),
        };
        assert!(
            !result.contains(&not_expected),
            "Did not expect {:?} to be in {:?}",
            not_expected,
            result
        );
        Ok(())
    }
    #[test]
    fn test_remove_fact() -> Result<(), Box<dyn std::error::Error>> {
        let mut network = ReteNetwork::new();
        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(":type".to_string()),
                object: Term::Constant("human".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(":type".to_string()),
                object: Term::Constant("mortal".to_string()),
            }],
        };
        network.add_rule(&rule)?;
        let initial_fact = RuleAtom::Triple {
            subject: Term::Constant("Socrates".to_string()),
            predicate: Term::Constant(":type".to_string()),
            object: Term::Constant("human".to_string()),
        };
        let facts = vec![initial_fact.clone()];
        let result = network.forward_chain(facts)?;
        let expected_derived = RuleAtom::Triple {
            subject: Term::Constant("Socrates".to_string()),
            predicate: Term::Constant(":type".to_string()),
            object: Term::Constant("mortal".to_string()),
        };
        assert!(
            result.contains(&initial_fact),
            "Expected initial fact {:?} to be in result {:?}",
            initial_fact,
            result
        );
        assert!(
            result.contains(&expected_derived),
            "Expected derived fact {:?} to be in result {:?}",
            expected_derived,
            result
        );
        let facts_before = network.get_facts();
        assert_eq!(
            facts_before.len(),
            1,
            "Should have 1 fact before removal: {:?}",
            facts_before
        );
        assert!(
            facts_before.contains(&initial_fact),
            "Should contain initial fact before removal"
        );
        let fact_to_remove = RuleAtom::Triple {
            subject: Term::Constant("Socrates".to_string()),
            predicate: Term::Constant(":type".to_string()),
            object: Term::Constant("human".to_string()),
        };
        let remove_result = network.remove_fact(&fact_to_remove);
        assert!(
            remove_result.is_ok(),
            "remove_fact should succeed: {:?}",
            remove_result
        );
        let facts_after = network.get_facts();
        assert_eq!(
            facts_after.len(),
            0,
            "Should have 0 facts after removal, but got: {:?}",
            facts_after
        );
        assert!(
            !facts_after.contains(&initial_fact),
            "Should not contain initial fact after removal"
        );
        Ok(())
    }
    #[test]
    fn test_remove_fact_with_multiple_patterns() -> Result<(), Box<dyn std::error::Error>> {
        let mut network = ReteNetwork::new();
        let rule1 = Rule {
            name: "rule1".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(":type".to_string()),
                object: Term::Constant("human".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(":category".to_string()),
                object: Term::Constant("person".to_string()),
            }],
        };
        let rule2 = Rule {
            name: "rule2".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("Y".to_string()),
                predicate: Term::Constant(":type".to_string()),
                object: Term::Constant("human".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Y".to_string()),
                predicate: Term::Constant(":status".to_string()),
                object: Term::Constant("alive".to_string()),
            }],
        };
        network.add_rule(&rule1)?;
        network.add_rule(&rule2)?;
        let fact = RuleAtom::Triple {
            subject: Term::Constant("Alice".to_string()),
            predicate: Term::Constant(":type".to_string()),
            object: Term::Constant("human".to_string()),
        };
        let result = network.forward_chain(vec![fact.clone()])?;
        assert_eq!(result.len(), 3, "Should have 3 facts after forward chain");
        let facts_before = network.get_facts();
        assert!(
            facts_before.contains(&fact),
            "Should contain fact before removal"
        );
        network.remove_fact(&fact)?;
        let facts_after = network.get_facts();
        assert!(
            !facts_after.contains(&fact),
            "Should not contain fact after removal"
        );
        Ok(())
    }
    /// Regression: a two-atom rule body whose first atom is an `rdf:type`
    /// triple (full RDF IRI) must still fire through `add_fact`.
    ///
    /// Previously `analyze_join_conditions` attached a `type_check` builtin
    /// join condition, but `evaluate_builtin` had no arm for it and returned
    /// `Ok(false)`, so the beta join silently rejected every match and no fact
    /// was ever derived for such rules (the dominant shape in RDFS/OWL-RL).
    #[test]
    fn regression_rete_rdf_type_join_fires() -> Result<(), Box<dyn std::error::Error>> {
        const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
        const SUBCLASS_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
        let mut network = ReteNetwork::new();
        // ?x rdf:type ?c , ?c rdfs:subClassOf ?d  =>  ?x rdf:type ?d
        let rule = Rule {
            name: "rdfs-subclass".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("x".to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Variable("c".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("c".to_string()),
                    predicate: Term::Constant(SUBCLASS_OF.to_string()),
                    object: Term::Variable("d".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("x".to_string()),
                predicate: Term::Constant(RDF_TYPE.to_string()),
                object: Term::Variable("d".to_string()),
            }],
        };
        network.add_rule(&rule)?;
        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("fido".to_string()),
                predicate: Term::Constant(RDF_TYPE.to_string()),
                object: Term::Constant("Dog".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("Dog".to_string()),
                predicate: Term::Constant(SUBCLASS_OF.to_string()),
                object: Term::Constant("Animal".to_string()),
            },
        ];
        let derived = network.forward_chain(facts)?;
        let expected = RuleAtom::Triple {
            subject: Term::Constant("fido".to_string()),
            predicate: Term::Constant(RDF_TYPE.to_string()),
            object: Term::Constant("Animal".to_string()),
        };
        assert!(
            derived.contains(&expected),
            "rdf:type-based two-atom rule must fire; got: {derived:?}"
        );
        Ok(())
    }
    /// Regression: after `clear()` the network must remain reusable AND continue
    /// to enforce compiled comparison filters. Previously `clear()` dropped
    /// `enhanced_beta_nodes`, so a clear()-then-reuse cycle fell onto the
    /// fallback join whose `apply_filter` returned `true` for every filter —
    /// silently deriving facts that violate the rule's `>` guard.
    #[test]
    fn regression_rete_clear_preserves_filters() -> Result<(), Box<dyn std::error::Error>> {
        let mut network = ReteNetwork::new();
        // ?person :hasAge ?age , ?age > 18  =>  ?person :isAdult true
        let rule = Rule {
            name: "isAdult".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("person".to_string()),
                    predicate: Term::Constant(":hasAge".to_string()),
                    object: Term::Variable("age".to_string()),
                },
                RuleAtom::GreaterThan {
                    left: Term::Variable("age".to_string()),
                    right: Term::Literal("18".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("person".to_string()),
                predicate: Term::Constant(":isAdult".to_string()),
                object: Term::Literal("true".to_string()),
            }],
        };
        network.add_rule(&rule)?;
        // First use.
        network.forward_chain(vec![RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant(":hasAge".to_string()),
            object: Term::Literal("20".to_string()),
        }])?;

        // Reset facts, keep compiled rules, then reuse.
        network.clear();

        let reuse = network.forward_chain(vec![
            RuleAtom::Triple {
                subject: Term::Constant("mary".to_string()),
                predicate: Term::Constant(":hasAge".to_string()),
                object: Term::Literal("10".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("bob".to_string()),
                predicate: Term::Constant(":hasAge".to_string()),
                object: Term::Literal("30".to_string()),
            },
        ])?;

        let adult_bob = RuleAtom::Triple {
            subject: Term::Constant("bob".to_string()),
            predicate: Term::Constant(":isAdult".to_string()),
            object: Term::Literal("true".to_string()),
        };
        let adult_mary = RuleAtom::Triple {
            subject: Term::Constant("mary".to_string()),
            predicate: Term::Constant(":isAdult".to_string()),
            object: Term::Literal("true".to_string()),
        };
        assert!(
            reuse.contains(&adult_bob),
            "bob (age 30) must satisfy age > 18 after clear+reuse; got: {reuse:?}"
        );
        assert!(
            !reuse.contains(&adult_mary),
            "mary (age 10) must NOT satisfy age > 18 — filter must survive clear(); got: {reuse:?}"
        );
        Ok(())
    }
}
