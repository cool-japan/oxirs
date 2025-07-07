//! # OxiRS Rule Engine
//!
//! Forward/backward rule engine for RDFS, OWL, and SWRL reasoning.
//!
//! This crate provides rule-based reasoning capabilities for knowledge graphs,
//! supporting both forward and backward chaining inference.

use anyhow::Result;

pub mod backward;
pub mod cache;
pub mod comprehensive_tutorial;
pub mod debug;
pub mod forward;
pub mod getting_started;
pub mod integration;
pub mod owl;
pub mod performance;
pub mod rdf_integration;
pub mod rdf_processing_simple;
pub mod rdfs;
pub mod rete;
pub mod rete_enhanced;
pub mod swrl;

/// Rule representation
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub body: Vec<RuleAtom>,
    pub head: Vec<RuleAtom>,
}

/// Rule atom (triple pattern or builtin)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum RuleAtom {
    Triple {
        subject: Term,
        predicate: Term,
        object: Term,
    },
    Builtin {
        name: String,
        args: Vec<Term>,
    },
    NotEqual {
        left: Term,
        right: Term,
    },
    GreaterThan {
        left: Term,
        right: Term,
    },
    LessThan {
        left: Term,
        right: Term,
    },
}

/// Rule term
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Term {
    Variable(String),
    Constant(String),
    Literal(String),
    Function { name: String, args: Vec<Term> },
}

/// Integrated rule engine combining all reasoning modes
#[derive(Debug)]
pub struct RuleEngine {
    rules: Vec<Rule>,
    forward_chainer: forward::ForwardChainer,
    backward_chainer: backward::BackwardChainer,
    rete_network: rete::ReteNetwork,
}

impl RuleEngine {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            forward_chainer: forward::ForwardChainer::new(),
            backward_chainer: backward::BackwardChainer::new(),
            rete_network: rete::ReteNetwork::new(),
        }
    }

    /// Add a rule to the engine
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule.clone());
        self.forward_chainer.add_rule(rule.clone());
        self.backward_chainer.add_rule(rule.clone());
        let _ = self.rete_network.add_rule(&rule);
    }

    /// Add multiple rules
    pub fn add_rules(&mut self, rules: Vec<Rule>) {
        for rule in rules {
            self.add_rule(rule);
        }
    }

    /// Add facts to the knowledge base
    pub fn add_facts(&mut self, facts: Vec<RuleAtom>) {
        self.forward_chainer.add_facts(facts.clone());
        self.backward_chainer.add_facts(facts);
    }

    /// Perform forward chaining inference
    pub fn forward_chain(&mut self, facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        self.forward_chainer.add_facts(facts.to_vec());
        self.forward_chainer.infer()
    }

    /// Perform backward chaining to prove a goal
    pub fn backward_chain(&mut self, goal: &RuleAtom) -> Result<bool> {
        self.backward_chainer.prove(goal)
    }

    /// Perform RETE-based forward chaining
    pub fn rete_forward_chain(&mut self, facts: Vec<RuleAtom>) -> Result<Vec<RuleAtom>> {
        self.rete_network.forward_chain(facts)
    }

    /// Get all current facts
    pub fn get_facts(&self) -> Vec<RuleAtom> {
        self.forward_chainer.get_facts()
    }

    /// Clear all facts and caches
    pub fn clear(&mut self) {
        self.forward_chainer.clear_facts();
        self.backward_chainer.clear_facts();
        self.rete_network.clear();
    }

    /// Add a single fact to the knowledge base
    pub fn add_fact(&mut self, fact: RuleAtom) {
        self.add_facts(vec![fact]);
    }

    /// Set cache (placeholder implementation)
    pub fn set_cache(&mut self, _cache: Option<crate::cache::RuleCache>) {
        // TODO: Implement cache setting
    }

    /// Get cache (placeholder implementation)
    pub fn get_cache(&self) -> Option<crate::cache::RuleCache> {
        // TODO: Implement cache getting
        None
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod comprehensive_tests {
    use super::*;
    use crate::owl::OwlReasoner;
    use crate::rdfs::RdfsReasoner;
    use crate::swrl::{SwrlArgument, SwrlAtom, SwrlEngine, SwrlRule};

    /// Test integrated forward and backward chaining
    #[test]
    fn test_integrated_forward_backward_chaining() {
        let mut engine = RuleEngine::new();

        // Add simple inheritance rule: ancestor(X,Y) :- parent(X,Y)
        engine.add_rule(Rule {
            name: "simple_ancestor".to_string(),
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
        });

        // Add facts
        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("parent".to_string()),
            object: Term::Constant("mary".to_string()),
        }];

        // Test forward chaining
        let forward_results = engine.forward_chain(&facts).unwrap();

        // Should derive ancestor relationship from parent relationship
        assert!(forward_results.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "john" && p == "ancestor" && o == "mary")
        }));

        // Test backward chaining on a direct ancestor goal
        let goal = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("ancestor".to_string()),
            object: Term::Constant("mary".to_string()),
        };

        engine.add_facts(facts);
        assert!(engine.backward_chain(&goal).unwrap());
    }

    /// Test RDFS reasoning integration
    #[test]
    fn test_rdfs_integration() {
        let mut rdfs_reasoner = RdfsReasoner::new();

        // Add RDFS vocabulary
        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("Person".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string(),
                ),
                object: Term::Constant("LivingThing".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Constant("Person".to_string()),
            },
        ];

        let inferred = rdfs_reasoner.infer(&facts).unwrap();

        // Should infer that john is a LivingThing
        assert!(inferred.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "john" && p == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" && o == "LivingThing")
        }));
    }

    /// Test OWL reasoning integration
    #[test]
    fn test_owl_integration() {
        let mut owl_reasoner = OwlReasoner::new();

        // Add OWL equivalence
        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("Human".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/2002/07/owl#equivalentClass".to_string(),
                ),
                object: Term::Constant("Person".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Constant("Human".to_string()),
            },
        ];

        let inferred = owl_reasoner.infer(&facts).unwrap();

        // Should infer that john is a Person due to equivalence
        assert!(inferred.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "john" && p == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" && o == "Person")
        }));
    }

    /// Test SWRL reasoning integration
    #[test]
    fn test_swrl_integration() {
        let mut swrl_engine = SwrlEngine::new();

        // Create SWRL rule: Person(?x) ∧ hasAge(?x, ?age) ∧ greaterThan(?age, 18) → Adult(?x)
        let swrl_rule = SwrlRule {
            id: "adult_rule".to_string(),
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
                        SwrlArgument::Literal("18".to_string()),
                    ],
                },
            ],
            head: vec![SwrlAtom::Class {
                class_predicate: "Adult".to_string(),
                argument: SwrlArgument::Variable("x".to_string()),
            }],
            metadata: std::collections::HashMap::new(),
        };

        swrl_engine.add_rule(swrl_rule).unwrap();

        // Add facts
        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Constant("Person".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant("hasAge".to_string()),
                object: Term::Literal("25".to_string()),
            },
        ];

        let results = swrl_engine.execute(&facts).unwrap();

        // Should infer that john is an Adult
        assert!(results.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "john" && p == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" && o == "Adult")
        }));
    }

    /// Test RETE network integration
    #[test]
    fn test_rete_integration() {
        let mut engine = RuleEngine::new();

        // Add rules for testing RETE efficiency
        engine.add_rule(Rule {
            name: "rule1".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Person".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Human".to_string()),
            }],
        });

        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("alice".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Person".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("bob".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Person".to_string()),
            },
        ];

        let rete_results = engine.rete_forward_chain(facts).unwrap();

        // Should derive Human types for both alice and bob
        assert!(rete_results.len() >= 4); // Original facts + derived facts
        assert!(rete_results.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "alice" && p == "type" && o == "Human")
        }));
    }

    /// Test error handling and edge cases
    #[test]
    fn test_error_handling() {
        let mut engine = RuleEngine::new();

        // Test with empty facts
        let empty_facts = vec![];
        let results = engine.forward_chain(&empty_facts).unwrap();
        assert!(results.is_empty());

        // Test backward chaining with non-existent goal
        let non_existent_goal = RuleAtom::Triple {
            subject: Term::Constant("nonexistent".to_string()),
            predicate: Term::Constant("nonexistent".to_string()),
            object: Term::Constant("nonexistent".to_string()),
        };
        assert!(!engine.backward_chain(&non_existent_goal).unwrap());

        // Test with circular rules (should not cause infinite loops)
        engine.add_rule(Rule {
            name: "circular1".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("b".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("a".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        });

        engine.add_rule(Rule {
            name: "circular2".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("a".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("b".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        });

        let circular_facts = vec![RuleAtom::Triple {
            subject: Term::Constant("x".to_string()),
            predicate: Term::Constant("a".to_string()),
            object: Term::Constant("y".to_string()),
        }];

        // Should terminate without infinite loop
        let results = engine.forward_chain(&circular_facts).unwrap();
        assert!(!results.is_empty());
    }

    /// Test performance with large knowledge bases
    #[test]
    fn test_performance_scalability() {
        let mut engine = RuleEngine::new();

        // Add a simple rule
        engine.add_rule(Rule {
            name: "simple_rule".to_string(),
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

        // Generate large number of facts
        let mut large_facts = Vec::new();
        for i in 0..1000 {
            large_facts.push(RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{i}")),
                predicate: Term::Constant("input".to_string()),
                object: Term::Constant(format!("value_{i}")),
            });
        }

        // Test that it can handle large inputs efficiently
        let start = std::time::Instant::now();
        let results = engine.forward_chain(&large_facts).unwrap();
        let duration = start.elapsed();

        // Should complete in reasonable time (< 1 second for 1000 facts)
        assert!(duration.as_secs() < 1);
        assert!(results.len() >= 2000); // Input + output facts
    }

    /// Test cross-reasoning compatibility
    #[test]
    fn test_cross_reasoning_compatibility() {
        let mut engine = RuleEngine::new();
        let mut rdfs_reasoner = RdfsReasoner::new();
        let mut owl_reasoner = OwlReasoner::new();

        // Start with RDFS facts
        let rdfs_facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("Student".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string(),
                ),
                object: Term::Constant("Person".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Constant("Student".to_string()),
            },
        ];

        // Apply RDFS reasoning
        let rdfs_inferred = rdfs_reasoner.infer(&rdfs_facts).unwrap();

        // Apply OWL reasoning to RDFS results
        let owl_inferred = owl_reasoner.infer(&rdfs_inferred).unwrap();

        // Apply rule engine to combined results
        engine.add_rule(Rule {
            name: "person_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Constant("Person".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Constant("conscious".to_string()),
            }],
        });

        let final_results = engine.forward_chain(&owl_inferred).unwrap();

        // Should derive that john is conscious through the chain:
        // john:Student -> john:Person (RDFS) -> john:conscious (rules)
        assert!(final_results.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "john" && p == "hasProperty" && o == "conscious")
        }));
    }

    /// Test complex rule interactions
    #[test]
    fn test_complex_rule_interactions() {
        let mut engine = RuleEngine::new();

        // Add complex interacting rules
        engine.add_rules(vec![
            Rule {
                name: "rule1".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("hasParent".to_string()),
                    object: Term::Variable("Y".to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("hasChild".to_string()),
                    object: Term::Variable("X".to_string()),
                }],
            },
            Rule {
                name: "rule2".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("hasChild".to_string()),
                    object: Term::Variable("Y".to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("isParent".to_string()),
                    object: Term::Constant("true".to_string()),
                }],
            },
            Rule {
                name: "rule3".to_string(),
                body: vec![
                    RuleAtom::Triple {
                        subject: Term::Variable("X".to_string()),
                        predicate: Term::Constant("isParent".to_string()),
                        object: Term::Constant("true".to_string()),
                    },
                    RuleAtom::Triple {
                        subject: Term::Variable("X".to_string()),
                        predicate: Term::Constant("age".to_string()),
                        object: Term::Variable("A".to_string()),
                    },
                ],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("category".to_string()),
                    object: Term::Constant("adult_parent".to_string()),
                }],
            },
        ]);

        let complex_facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("mary".to_string()),
                predicate: Term::Constant("hasParent".to_string()),
                object: Term::Constant("john".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant("age".to_string()),
                object: Term::Constant("45".to_string()),
            },
        ];

        let results = engine.forward_chain(&complex_facts).unwrap();

        // Should derive through chain of rules:
        // mary hasParent john -> john hasChild mary -> john isParent true -> john category adult_parent
        assert!(results.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "john" && p == "category" && o == "adult_parent")
        }));
    }
}
