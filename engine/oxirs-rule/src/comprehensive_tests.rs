use super::*;
use crate::owl::OwlReasoner;
use crate::rdfs::RdfsReasoner;
use crate::swrl::{SwrlArgument, SwrlAtom, SwrlEngine, SwrlRule};

/// Test integrated forward and backward chaining
#[test]
fn test_integrated_forward_backward_chaining() -> Result<(), Box<dyn std::error::Error>> {
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
    let forward_results = engine.forward_chain(&facts)?;

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
    assert!(engine.backward_chain(&goal)?);
    Ok(())
}

/// Test RDFS reasoning integration
#[test]
fn test_rdfs_integration() -> Result<(), Box<dyn std::error::Error>> {
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

    let inferred = rdfs_reasoner.infer(&facts)?;

    // Should infer that john is a LivingThing
    assert!(inferred.iter().any(|fact| {
        matches!(fact, RuleAtom::Triple {
            subject: Term::Constant(s),
            predicate: Term::Constant(p),
            object: Term::Constant(o)
        } if s == "john" && p == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" && o == "LivingThing")
    }));
    Ok(())
}

/// Test OWL reasoning integration
#[test]
fn test_owl_integration() -> Result<(), Box<dyn std::error::Error>> {
    let mut owl_reasoner = OwlReasoner::new();

    // Add OWL equivalence
    let facts = vec![
        RuleAtom::Triple {
            subject: Term::Constant("Human".to_string()),
            predicate: Term::Constant("http://www.w3.org/2002/07/owl#equivalentClass".to_string()),
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

    let inferred = owl_reasoner.infer(&facts)?;

    // Should infer that john is a Person due to equivalence
    assert!(inferred.iter().any(|fact| {
        matches!(fact, RuleAtom::Triple {
            subject: Term::Constant(s),
            predicate: Term::Constant(p),
            object: Term::Constant(o)
        } if s == "john" && p == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" && o == "Person")
    }));
    Ok(())
}

/// Test SWRL reasoning integration
#[test]
fn test_swrl_integration() -> Result<(), Box<dyn std::error::Error>> {
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

    swrl_engine.add_rule(swrl_rule)?;

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

    let results = swrl_engine.execute(&facts)?;

    // Should infer that john is an Adult
    assert!(results.iter().any(|fact| {
        matches!(fact, RuleAtom::Triple {
            subject: Term::Constant(s),
            predicate: Term::Constant(p),
            object: Term::Constant(o)
        } if s == "john" && p == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" && o == "Adult")
    }));
    Ok(())
}

/// Test RETE network integration
#[test]
fn test_rete_integration() -> Result<(), Box<dyn std::error::Error>> {
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

    let rete_results = engine.rete_forward_chain(facts)?;

    // Should derive Human types for both alice and bob
    assert!(rete_results.len() >= 4); // Original facts + derived facts
    assert!(rete_results.iter().any(|fact| {
        matches!(fact, RuleAtom::Triple {
            subject: Term::Constant(s),
            predicate: Term::Constant(p),
            object: Term::Constant(o)
        } if s == "alice" && p == "type" && o == "Human")
    }));
    Ok(())
}

/// Test error handling and edge cases
#[test]
fn test_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = RuleEngine::new();

    // Test with empty facts
    let empty_facts = vec![];
    let results = engine.forward_chain(&empty_facts)?;
    assert!(results.is_empty());

    // Test backward chaining with non-existent goal
    let non_existent_goal = RuleAtom::Triple {
        subject: Term::Constant("nonexistent".to_string()),
        predicate: Term::Constant("nonexistent".to_string()),
        object: Term::Constant("nonexistent".to_string()),
    };
    assert!(!engine.backward_chain(&non_existent_goal)?);

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
    let results = engine.forward_chain(&circular_facts)?;
    assert!(!results.is_empty());
    Ok(())
}

/// Test performance with large knowledge bases
#[test]
fn test_performance_scalability() -> Result<(), Box<dyn std::error::Error>> {
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
    let results = engine.forward_chain(&large_facts)?;
    let duration = start.elapsed();

    // Should complete in reasonable time (< 1 second for 1000 facts)
    assert!(duration.as_secs() < 1);
    assert!(results.len() >= 2000); // Input + output facts
    Ok(())
}

/// Test cross-reasoning compatibility
#[test]
fn test_cross_reasoning_compatibility() -> Result<(), Box<dyn std::error::Error>> {
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
    let rdfs_inferred = rdfs_reasoner.infer(&rdfs_facts)?;

    // Apply OWL reasoning to RDFS results
    let owl_inferred = owl_reasoner.infer(&rdfs_inferred)?;

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

    let final_results = engine.forward_chain(&owl_inferred)?;

    // Should derive that john is conscious through the chain:
    // john:Student -> john:Person (RDFS) -> john:conscious (rules)
    assert!(final_results.iter().any(|fact| {
        matches!(fact, RuleAtom::Triple {
            subject: Term::Constant(s),
            predicate: Term::Constant(p),
            object: Term::Constant(o)
        } if s == "john" && p == "hasProperty" && o == "conscious")
    }));
    Ok(())
}

/// Test complex rule interactions
#[test]
fn test_complex_rule_interactions() -> Result<(), Box<dyn std::error::Error>> {
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

    let results = engine.forward_chain(&complex_facts)?;

    // Should derive through chain of rules:
    // mary hasParent john -> john hasChild mary -> john isParent true -> john category adult_parent
    assert!(results.iter().any(|fact| {
        matches!(fact, RuleAtom::Triple {
            subject: Term::Constant(s),
            predicate: Term::Constant(p),
            object: Term::Constant(o)
        } if s == "john" && p == "category" && o == "adult_parent")
    }));
    Ok(())
}

// =========================================================================
// OWL INFERENCE TESTS
// =========================================================================

/// Test RDFS subclass transitivity (rdfs11)
#[test]
fn test_rdfs_subclass_transitivity() {
    let mut rdfs_reasoner = RdfsReasoner::new();

    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    let rdfs_subclass = "http://www.w3.org/2000/01/rdf-schema#subClassOf";

    // Animal -> LivingThing -> Entity  (transitive chain)
    let facts = vec![
        RuleAtom::Triple {
            subject: Term::Constant("Animal".to_string()),
            predicate: Term::Constant(rdfs_subclass.to_string()),
            object: Term::Constant("LivingThing".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("LivingThing".to_string()),
            predicate: Term::Constant(rdfs_subclass.to_string()),
            object: Term::Constant("Entity".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("cat".to_string()),
            predicate: Term::Constant(rdf_type.to_string()),
            object: Term::Constant("Animal".to_string()),
        },
    ];

    let inferred = rdfs_reasoner
        .infer(&facts)
        .expect("RDFS inference should succeed");

    // cat must be typed as LivingThing via rdfs9
    assert!(
        inferred.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "cat" && p == rdf_type && o == "LivingThing")
        }),
        "cat should be inferred as LivingThing"
    );

    // Animal must be a subclass of Entity via rdfs11 (transitive subclass)
    // Check the RDFS context which maintains transitive closure of class hierarchy
    assert!(
        rdfs_reasoner.context.is_subclass_of("Animal", "Entity"),
        "Animal should be transitive subClassOf Entity in RDFS context"
    );

    // Also ensure the inferred set is non-empty (inference ran)
    assert!(!inferred.is_empty(), "RDFS inference should produce facts");
}

/// Test RDFS domain inference (rdfs2)
#[test]
fn test_rdfs_domain_inference() {
    let mut rdfs_reasoner = RdfsReasoner::new();

    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    let rdfs_domain = "http://www.w3.org/2000/01/rdf-schema#domain";

    let facts = vec![
        // hasMother has domain Person
        RuleAtom::Triple {
            subject: Term::Constant("hasMother".to_string()),
            predicate: Term::Constant(rdfs_domain.to_string()),
            object: Term::Constant("Person".to_string()),
        },
        // alice hasMother eve — so alice must be a Person
        RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant("hasMother".to_string()),
            object: Term::Constant("eve".to_string()),
        },
    ];

    let inferred = rdfs_reasoner
        .infer(&facts)
        .expect("RDFS domain inference should succeed");

    // alice should be inferred as Person
    assert!(
        inferred.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "alice" && p == rdf_type && o == "Person")
        }),
        "alice should be inferred as Person via domain"
    );
}

/// Test RDFS range inference (rdfs3)
#[test]
fn test_rdfs_range_inference() {
    let mut rdfs_reasoner = RdfsReasoner::new();

    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    let rdfs_range = "http://www.w3.org/2000/01/rdf-schema#range";

    let facts = vec![
        // hasChild has range Person
        RuleAtom::Triple {
            subject: Term::Constant("hasChild".to_string()),
            predicate: Term::Constant(rdfs_range.to_string()),
            object: Term::Constant("Person".to_string()),
        },
        // john hasChild bob — so bob must be a Person
        RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("hasChild".to_string()),
            object: Term::Constant("bob".to_string()),
        },
    ];

    let inferred = rdfs_reasoner
        .infer(&facts)
        .expect("RDFS range inference should succeed");

    // bob should be inferred as Person
    assert!(
        inferred.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "bob" && p == rdf_type && o == "Person")
        }),
        "bob should be inferred as Person via range"
    );
}

/// Test RDFS subproperty transitivity (rdfs5)
#[test]
fn test_rdfs_subproperty_transitivity() {
    let mut rdfs_reasoner = RdfsReasoner::new();

    let rdfs_subproperty = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf";

    let facts = vec![
        // hasFather subPropertyOf hasParent, hasParent subPropertyOf hasAncestor
        RuleAtom::Triple {
            subject: Term::Constant("hasFather".to_string()),
            predicate: Term::Constant(rdfs_subproperty.to_string()),
            object: Term::Constant("hasParent".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("hasParent".to_string()),
            predicate: Term::Constant(rdfs_subproperty.to_string()),
            object: Term::Constant("hasAncestor".to_string()),
        },
    ];

    let inferred = rdfs_reasoner
        .infer(&facts)
        .expect("RDFS subproperty inference should succeed");

    // hasFather should be transitive subPropertyOf hasAncestor
    // The RDFS context maintains transitive closure of property hierarchy
    assert!(
        rdfs_reasoner
            .context
            .is_subproperty_of("hasFather", "hasAncestor"),
        "hasFather should be transitive subPropertyOf hasAncestor in RDFS context"
    );

    // Also ensure the inferred set is non-empty (inference ran)
    assert!(!inferred.is_empty(), "RDFS inference should produce facts");
}

/// Test RDFS subproperty inference (rdfs7)
#[test]
fn test_rdfs_subproperty_inference() {
    let mut rdfs_reasoner = RdfsReasoner::new();

    let rdfs_subproperty = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf";

    let facts = vec![
        // hasFather subPropertyOf hasParent
        RuleAtom::Triple {
            subject: Term::Constant("hasFather".to_string()),
            predicate: Term::Constant(rdfs_subproperty.to_string()),
            object: Term::Constant("hasParent".to_string()),
        },
        // alice hasFather bob — so alice hasParent bob must be inferred
        RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant("hasFather".to_string()),
            object: Term::Constant("bob".to_string()),
        },
    ];

    let inferred = rdfs_reasoner
        .infer(&facts)
        .expect("RDFS subproperty inference should succeed");

    // alice hasParent bob should be inferred via rdfs7
    assert!(
        inferred.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "alice" && p == "hasParent" && o == "bob")
        }),
        "alice hasParent bob should be inferred via subproperty rule"
    );
}

/// Test OWL transitive property inference
#[test]
fn test_owl_transitive_property() {
    let mut owl_reasoner = OwlReasoner::new();

    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    let owl_transitive = "http://www.w3.org/2002/07/owl#TransitiveProperty";

    let facts = vec![
        // locatedIn is transitive
        RuleAtom::Triple {
            subject: Term::Constant("locatedIn".to_string()),
            predicate: Term::Constant(rdf_type.to_string()),
            object: Term::Constant(owl_transitive.to_string()),
        },
        // Paris locatedIn France, France locatedIn Europe
        RuleAtom::Triple {
            subject: Term::Constant("Paris".to_string()),
            predicate: Term::Constant("locatedIn".to_string()),
            object: Term::Constant("France".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("France".to_string()),
            predicate: Term::Constant("locatedIn".to_string()),
            object: Term::Constant("Europe".to_string()),
        },
    ];

    let inferred = owl_reasoner
        .infer(&facts)
        .expect("OWL transitive inference should succeed");

    // Paris locatedIn Europe should be inferred transitively
    assert!(
        inferred.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "Paris" && p == "locatedIn" && o == "Europe")
        }),
        "Paris locatedIn Europe should be inferred via transitive property"
    );
}

/// Test OWL symmetric property inference
#[test]
fn test_owl_symmetric_property() {
    let mut owl_reasoner = OwlReasoner::new();

    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    let owl_symmetric = "http://www.w3.org/2002/07/owl#SymmetricProperty";

    let facts = vec![
        // isFriendOf is symmetric
        RuleAtom::Triple {
            subject: Term::Constant("isFriendOf".to_string()),
            predicate: Term::Constant(rdf_type.to_string()),
            object: Term::Constant(owl_symmetric.to_string()),
        },
        // alice isFriendOf bob — so bob isFriendOf alice should be inferred
        RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant("isFriendOf".to_string()),
            object: Term::Constant("bob".to_string()),
        },
    ];

    let inferred = owl_reasoner
        .infer(&facts)
        .expect("OWL symmetric inference should succeed");

    // bob isFriendOf alice should be inferred
    assert!(
        inferred.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "bob" && p == "isFriendOf" && o == "alice")
        }),
        "bob isFriendOf alice should be inferred via symmetric property"
    );
}

/// Test OWL inverse property inference
#[test]
fn test_owl_inverse_property() {
    let mut owl_reasoner = OwlReasoner::new();

    let owl_inverse_of = "http://www.w3.org/2002/07/owl#inverseOf";

    let facts = vec![
        // isParentOf inverseOf isChildOf
        RuleAtom::Triple {
            subject: Term::Constant("isParentOf".to_string()),
            predicate: Term::Constant(owl_inverse_of.to_string()),
            object: Term::Constant("isChildOf".to_string()),
        },
        // alice isParentOf bob — so bob isChildOf alice should be inferred
        RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant("isParentOf".to_string()),
            object: Term::Constant("bob".to_string()),
        },
    ];

    let inferred = owl_reasoner
        .infer(&facts)
        .expect("OWL inverse inference should succeed");

    // bob isChildOf alice should be inferred
    assert!(
        inferred.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "bob" && p == "isChildOf" && o == "alice")
        }),
        "bob isChildOf alice should be inferred via inverse property"
    );
}

/// Test OWL sameAs transitivity
#[test]
fn test_owl_sameas_transitivity() {
    let mut owl_reasoner = OwlReasoner::new();

    let owl_same_as = "http://www.w3.org/2002/07/owl#sameAs";

    let facts = vec![
        // A sameAs B, B sameAs C — C should be sameAs A via transitivity
        RuleAtom::Triple {
            subject: Term::Constant("ResourceA".to_string()),
            predicate: Term::Constant(owl_same_as.to_string()),
            object: Term::Constant("ResourceB".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("ResourceB".to_string()),
            predicate: Term::Constant(owl_same_as.to_string()),
            object: Term::Constant("ResourceC".to_string()),
        },
    ];

    let inferred = owl_reasoner
        .infer(&facts)
        .expect("OWL sameAs inference should succeed");

    // ResourceA sameAs ResourceC should be inferred
    assert!(
        inferred.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "ResourceA" && p == owl_same_as && o == "ResourceC")
        }),
        "ResourceA sameAs ResourceC should be inferred via transitivity"
    );
}

/// Test OWL equivalent class with deep hierarchy
#[test]
fn test_owl_equivalent_class_deep() {
    let mut owl_reasoner = OwlReasoner::new();

    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    let owl_equiv_class = "http://www.w3.org/2002/07/owl#equivalentClass";

    let facts = vec![
        // Human equivalentClass Person
        RuleAtom::Triple {
            subject: Term::Constant("Human".to_string()),
            predicate: Term::Constant(owl_equiv_class.to_string()),
            object: Term::Constant("Person".to_string()),
        },
        // Person equivalentClass Homo sapiens
        RuleAtom::Triple {
            subject: Term::Constant("Person".to_string()),
            predicate: Term::Constant(owl_equiv_class.to_string()),
            object: Term::Constant("HomoSapiens".to_string()),
        },
        // alice is a Human
        RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant(rdf_type.to_string()),
            object: Term::Constant("Human".to_string()),
        },
    ];

    let inferred = owl_reasoner
        .infer(&facts)
        .expect("OWL equivalent class inference should succeed");

    // alice should be inferred as Person
    assert!(
        inferred.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "alice" && p == rdf_type && o == "Person")
        }),
        "alice should be inferred as Person via equivalent class"
    );
}

// =========================================================================
// SWRL RULE EXECUTION TESTS
// =========================================================================

/// Test SWRL rule with less-than builtin
#[test]
fn test_swrl_less_than_builtin() -> Result<(), Box<dyn std::error::Error>> {
    let mut swrl_engine = SwrlEngine::new();

    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

    // Rule: Person(?x) ∧ hasAge(?x, ?age) ∧ lessThan(?age, 18) → Minor(?x)
    let swrl_rule = SwrlRule {
        id: "minor_rule".to_string(),
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
                builtin_predicate: "http://www.w3.org/2003/11/swrlb#lessThan".to_string(),
                arguments: vec![
                    SwrlArgument::Variable("age".to_string()),
                    SwrlArgument::Literal("18".to_string()),
                ],
            },
        ],
        head: vec![SwrlAtom::Class {
            class_predicate: "Minor".to_string(),
            argument: SwrlArgument::Variable("x".to_string()),
        }],
        metadata: std::collections::HashMap::new(),
    };

    swrl_engine
        .add_rule(swrl_rule)
        .expect("SWRL rule should be added");

    let facts = vec![
        RuleAtom::Triple {
            subject: Term::Constant("tommy".to_string()),
            predicate: Term::Constant(rdf_type.to_string()),
            object: Term::Constant("Person".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("tommy".to_string()),
            predicate: Term::Constant("hasAge".to_string()),
            object: Term::Literal("12".to_string()),
        },
    ];

    let results = swrl_engine
        .execute(&facts)
        .expect("SWRL execution should succeed");

    assert!(
        results.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "tommy" && p == rdf_type && o == "Minor")
        }),
        "tommy should be inferred as Minor via SWRL lessThan builtin"
    );
    Ok(())
}

/// Test SWRL rule with individual property atom
#[test]
fn test_swrl_individual_property_rule() -> Result<(), Box<dyn std::error::Error>> {
    let mut swrl_engine = SwrlEngine::new();

    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

    // Rule: Doctor(?x) ∧ worksAt(?x, ?hospital) ∧ Hospital(?hospital) → MedicalProfessional(?x)
    let swrl_rule = SwrlRule {
        id: "medical_professional_rule".to_string(),
        body: vec![
            SwrlAtom::Class {
                class_predicate: "Doctor".to_string(),
                argument: SwrlArgument::Variable("x".to_string()),
            },
            SwrlAtom::IndividualProperty {
                property_predicate: "worksAt".to_string(),
                argument1: SwrlArgument::Variable("x".to_string()),
                argument2: SwrlArgument::Variable("h".to_string()),
            },
            SwrlAtom::Class {
                class_predicate: "Hospital".to_string(),
                argument: SwrlArgument::Variable("h".to_string()),
            },
        ],
        head: vec![SwrlAtom::Class {
            class_predicate: "MedicalProfessional".to_string(),
            argument: SwrlArgument::Variable("x".to_string()),
        }],
        metadata: std::collections::HashMap::new(),
    };

    swrl_engine
        .add_rule(swrl_rule)
        .expect("SWRL rule should be added");

    let facts = vec![
        RuleAtom::Triple {
            subject: Term::Constant("drsmith".to_string()),
            predicate: Term::Constant(rdf_type.to_string()),
            object: Term::Constant("Doctor".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("drsmith".to_string()),
            predicate: Term::Constant("worksAt".to_string()),
            object: Term::Constant("generalhosp".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("generalhosp".to_string()),
            predicate: Term::Constant(rdf_type.to_string()),
            object: Term::Constant("Hospital".to_string()),
        },
    ];

    let results = swrl_engine
        .execute(&facts)
        .expect("SWRL execution should succeed");

    assert!(
        results.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "drsmith" && p == rdf_type && o == "MedicalProfessional")
        }),
        "drsmith should be inferred as MedicalProfessional"
    );
    Ok(())
}

/// Test SWRL rule with same individual constraint
#[test]
fn test_swrl_same_individual_constraint() -> Result<(), Box<dyn std::error::Error>> {
    let mut swrl_engine = SwrlEngine::new();

    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

    // Rule: Person(?x) ∧ sameAs(?x, ?y) → Person(?y)
    let swrl_rule = SwrlRule {
        id: "same_class_rule".to_string(),
        body: vec![
            SwrlAtom::Class {
                class_predicate: "Person".to_string(),
                argument: SwrlArgument::Variable("x".to_string()),
            },
            SwrlAtom::SameIndividual {
                argument1: SwrlArgument::Variable("x".to_string()),
                argument2: SwrlArgument::Variable("y".to_string()),
            },
        ],
        head: vec![SwrlAtom::Class {
            class_predicate: "Person".to_string(),
            argument: SwrlArgument::Variable("y".to_string()),
        }],
        metadata: std::collections::HashMap::new(),
    };

    swrl_engine
        .add_rule(swrl_rule)
        .expect("SWRL rule should be added");

    let owl_same_as = "http://www.w3.org/2002/07/owl#sameAs";

    let facts = vec![
        RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant(rdf_type.to_string()),
            object: Term::Constant("Person".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant(owl_same_as.to_string()),
            object: Term::Constant("alicia".to_string()),
        },
    ];

    let results = swrl_engine
        .execute(&facts)
        .expect("SWRL execution should succeed");

    assert!(
        results.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "alicia" && p == rdf_type && o == "Person")
        }),
        "alicia should be inferred as Person via sameAs"
    );
    Ok(())
}

// =========================================================================
// RULE CHAINING (A→B→C) TESTS
// =========================================================================

/// Test three-level rule chaining A→B→C
#[test]
fn test_three_level_rule_chain() {
    let mut engine = RuleEngine::new();

    // A→B: node(X) → processable(X)
    // B→C: processable(X) → indexed(X)
    // C→D: indexed(X) → searchable(X)
    engine.add_rules(vec![
        Rule {
            name: "node_to_processable".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Node".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("status".to_string()),
                object: Term::Constant("processable".to_string()),
            }],
        },
        Rule {
            name: "processable_to_indexed".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("status".to_string()),
                object: Term::Constant("processable".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("status".to_string()),
                object: Term::Constant("indexed".to_string()),
            }],
        },
        Rule {
            name: "indexed_to_searchable".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("status".to_string()),
                object: Term::Constant("indexed".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("status".to_string()),
                object: Term::Constant("searchable".to_string()),
            }],
        },
    ]);

    let facts = vec![RuleAtom::Triple {
        subject: Term::Constant("doc1".to_string()),
        predicate: Term::Constant("type".to_string()),
        object: Term::Constant("Node".to_string()),
    }];

    let results = engine
        .forward_chain(&facts)
        .expect("Rule chaining should succeed");

    // doc1 should be searchable after chain: Node→processable→indexed→searchable
    assert!(
        results.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "doc1" && p == "status" && o == "searchable")
        }),
        "doc1 should be searchable via three-step rule chain"
    );
}

/// Test rule chain with multiple facts propagating through
#[test]
fn test_rule_chain_multiple_entities() {
    let mut engine = RuleEngine::new();

    // employee → worker → taxpayer
    engine.add_rules(vec![
        Rule {
            name: "employee_to_worker".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("role".to_string()),
                object: Term::Constant("employee".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("role".to_string()),
                object: Term::Constant("worker".to_string()),
            }],
        },
        Rule {
            name: "worker_to_taxpayer".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("role".to_string()),
                object: Term::Constant("worker".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("role".to_string()),
                object: Term::Constant("taxpayer".to_string()),
            }],
        },
    ]);

    let facts = vec![
        RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant("role".to_string()),
            object: Term::Constant("employee".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("bob".to_string()),
            predicate: Term::Constant("role".to_string()),
            object: Term::Constant("employee".to_string()),
        },
    ];

    let results = engine
        .forward_chain(&facts)
        .expect("Rule chain with multiple entities should succeed");

    // Both alice and bob should be taxpayers
    let alice_taxpayer = results.iter().any(|fact| {
        matches!(fact, RuleAtom::Triple {
            subject: Term::Constant(s),
            predicate: Term::Constant(p),
            object: Term::Constant(o)
        } if s == "alice" && p == "role" && o == "taxpayer")
    });

    let bob_taxpayer = results.iter().any(|fact| {
        matches!(fact, RuleAtom::Triple {
            subject: Term::Constant(s),
            predicate: Term::Constant(p),
            object: Term::Constant(o)
        } if s == "bob" && p == "role" && o == "taxpayer")
    });

    assert!(alice_taxpayer, "alice should be inferred as taxpayer");
    assert!(bob_taxpayer, "bob should be inferred as taxpayer");
}

/// Test rule chaining for classification
#[test]
fn test_classification_chain() {
    let mut engine = RuleEngine::new();

    // Scientist → Researcher → AcademicProfessional
    engine.add_rules(vec![
        Rule {
            name: "scientist_to_researcher".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Scientist".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Researcher".to_string()),
            }],
        },
        Rule {
            name: "researcher_to_academic".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Researcher".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("AcademicProfessional".to_string()),
            }],
        },
    ]);

    let facts = vec![RuleAtom::Triple {
        subject: Term::Constant("einstein".to_string()),
        predicate: Term::Constant("type".to_string()),
        object: Term::Constant("Scientist".to_string()),
    }];

    let results = engine
        .forward_chain(&facts)
        .expect("Classification chain should succeed");

    assert!(
        results.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "einstein" && p == "type" && o == "AcademicProfessional")
        }),
        "einstein should be inferred as AcademicProfessional via classification chain"
    );
}
