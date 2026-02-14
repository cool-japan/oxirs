//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::metrics::{Counter, Gauge};

lazy_static::lazy_static! {
    pub static ref TOKEN_CLONES : Counter = Counter::new("rete_token_clones".to_string());
    pub static ref ACTIVE_TOKENS : Gauge = Gauge::new("rete_active_tokens".to_string());
}
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
    fn test_simple_rule_compilation() {
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
        network.add_rule(&rule).unwrap();
        let stats = network.get_stats();
        assert!(stats.alpha_nodes > 0);
        assert!(stats.production_nodes > 0);
    }
    #[test]
    fn test_fact_processing() {
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
        network.add_rule(&rule).unwrap();
        let fact = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        };
        let derived = network.add_fact(fact).unwrap();
        assert!(!derived.is_empty());
        let expected = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("mortal".to_string()),
        };
        assert!(derived.contains(&expected));
    }
    #[test]
    fn test_forward_chaining() {
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
        network.add_rule(&rule).unwrap();
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
        let all_facts = network.forward_chain(initial_facts).unwrap();
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
    }
    #[test]
    fn test_enhanced_beta_join() {
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
        network.add_rule(&rule).unwrap();
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
        let results = network.forward_chain(facts).unwrap();
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
    }
    #[test]
    fn test_memory_management() {
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
        network.add_rule(&rule).unwrap();
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
        network.forward_chain(facts).unwrap();
        let stats = network.get_enhanced_stats();
        println!(
            "Memory management stats: memory_evictions={}, peak_memory_usage={}, enhanced_nodes={}",
            stats.memory_evictions, stats.peak_memory_usage, stats.enhanced_nodes
        );
        assert!(stats.memory_evictions > 0);
        assert!(stats.enhanced_nodes > 0);
    }
    #[test]
    fn test_conflict_resolution() {
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
        network.add_rule(&rule).unwrap();
        let fact = RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("p".to_string()),
            object: Term::Constant("b".to_string()),
        };
        let results = network.add_fact(fact).unwrap();
        assert!(!results.is_empty());
    }
    #[test]
    fn test_comparison_operator_greater_than() {
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
        network.add_rule(&rule).unwrap();
        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant(":hasAge".to_string()),
            object: Term::Literal("20".to_string()),
        }];
        let result = network.forward_chain(facts).unwrap();
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
    }
    #[test]
    fn test_comparison_operator_less_than() {
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
        network.add_rule(&rule).unwrap();
        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant(":hasAge".to_string()),
            object: Term::Literal("15".to_string()),
        }];
        let result = network.forward_chain(facts).unwrap();
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
    }
    #[test]
    fn test_comparison_operator_filter_fails() {
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
        network.add_rule(&rule).unwrap();
        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("bob".to_string()),
            predicate: Term::Constant(":hasAge".to_string()),
            object: Term::Literal("10".to_string()),
        }];
        let result = network.forward_chain(facts).unwrap();
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
    }
    #[test]
    fn test_remove_fact() {
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
        network.add_rule(&rule).unwrap();
        let initial_fact = RuleAtom::Triple {
            subject: Term::Constant("Socrates".to_string()),
            predicate: Term::Constant(":type".to_string()),
            object: Term::Constant("human".to_string()),
        };
        let facts = vec![initial_fact.clone()];
        let result = network.forward_chain(facts).unwrap();
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
    }
    #[test]
    fn test_remove_fact_with_multiple_patterns() {
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
        network.add_rule(&rule1).unwrap();
        network.add_rule(&rule2).unwrap();
        let fact = RuleAtom::Triple {
            subject: Term::Constant("Alice".to_string()),
            predicate: Term::Constant(":type".to_string()),
            object: Term::Constant("human".to_string()),
        };
        let result = network.forward_chain(vec![fact.clone()]).unwrap();
        assert_eq!(result.len(), 3, "Should have 3 facts after forward chain");
        let facts_before = network.get_facts();
        assert!(
            facts_before.contains(&fact),
            "Should contain fact before removal"
        );
        network.remove_fact(&fact).unwrap();
        let facts_after = network.get_facts();
        assert!(
            !facts_after.contains(&fact),
            "Should not contain fact after removal"
        );
    }
}
