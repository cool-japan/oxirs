//! Extended comprehensive tests — temporal reasoning, negation, probabilistic,
//! edge cases, performance, cache management, term/atom, and helper tests.
//!
//! Split from comprehensive_tests.rs to keep each file under 2000 lines.

use super::*;

// =========================================================================
// TEMPORAL REASONING TESTS
// =========================================================================

/// Test Allen relation Before
#[test]
fn test_temporal_allen_before_relation() {
    use crate::temporal::{AllenRelation, TimeInterval};

    let morning = TimeInterval::new(8.0, 12.0).expect("morning interval should be valid");
    let afternoon = TimeInterval::new(13.0, 17.0).expect("afternoon interval should be valid");

    assert_eq!(
        morning.allen_relation(&afternoon),
        AllenRelation::Before,
        "morning should be Before afternoon"
    );
    assert_eq!(
        afternoon.allen_relation(&morning),
        AllenRelation::After,
        "afternoon should be After morning"
    );
}

/// Test Allen relation Meets
#[test]
fn test_temporal_allen_meets_relation() {
    use crate::temporal::{AllenRelation, TimeInterval};

    let first = TimeInterval::new(8.0, 12.0).expect("first interval should be valid");
    let second = TimeInterval::new(12.0, 16.0).expect("second interval should be valid");

    assert_eq!(
        first.allen_relation(&second),
        AllenRelation::Meets,
        "first interval should Meet second"
    );
    assert_eq!(
        second.allen_relation(&first),
        AllenRelation::MetBy,
        "second should be MetBy first"
    );
}

/// Test Allen relation Overlaps
#[test]
fn test_temporal_allen_overlaps_relation() {
    use crate::temporal::{AllenRelation, TimeInterval};

    let first = TimeInterval::new(8.0, 14.0).expect("first interval should be valid");
    let second = TimeInterval::new(12.0, 18.0).expect("second interval should be valid");

    assert_eq!(
        first.allen_relation(&second),
        AllenRelation::Overlaps,
        "first should Overlap second"
    );
    assert_eq!(
        second.allen_relation(&first),
        AllenRelation::OverlappedBy,
        "second should be OverlappedBy first"
    );
}

/// Test Allen relation During/Contains
#[test]
fn test_temporal_allen_during_relation() {
    use crate::temporal::{AllenRelation, TimeInterval};

    let container = TimeInterval::new(8.0, 18.0).expect("container interval should be valid");
    let inner = TimeInterval::new(10.0, 14.0).expect("inner interval should be valid");

    assert_eq!(
        inner.allen_relation(&container),
        AllenRelation::During,
        "inner should be During container"
    );
    assert_eq!(
        container.allen_relation(&inner),
        AllenRelation::Contains,
        "container should Contain inner"
    );
}

/// Test Allen relation Equals
#[test]
fn test_temporal_allen_equals_relation() {
    use crate::temporal::{AllenRelation, TimeInterval};

    let interval_a = TimeInterval::new(10.0, 14.0).expect("interval_a should be valid");
    let interval_b = TimeInterval::new(10.0, 14.0).expect("interval_b should be valid");

    assert_eq!(
        interval_a.allen_relation(&interval_b),
        AllenRelation::Equals,
        "equal intervals should have Equals relation"
    );
}

/// Test interval intersection
#[test]
fn test_temporal_interval_intersection() {
    use crate::temporal::TimeInterval;

    let first = TimeInterval::new(8.0, 14.0).expect("first interval should be valid");
    let second = TimeInterval::new(12.0, 18.0).expect("second interval should be valid");

    let intersection = first
        .intersection(&second)
        .expect("should have intersection");
    assert_eq!(intersection.start, 12.0, "intersection should start at 12");
    assert_eq!(intersection.end, 14.0, "intersection should end at 14");
}

/// Test non-overlapping intervals have no intersection
#[test]
fn test_temporal_no_intersection() {
    use crate::temporal::TimeInterval;

    let first = TimeInterval::new(8.0, 10.0).expect("first interval should be valid");
    let second = TimeInterval::new(12.0, 14.0).expect("second interval should be valid");

    assert!(
        first.intersection(&second).is_none(),
        "non-overlapping intervals should have no intersection"
    );
}

/// Test Allen relation inverse property
#[test]
fn test_temporal_inverse_relations() {
    use crate::temporal::AllenRelation;

    assert_eq!(AllenRelation::Before.inverse(), AllenRelation::After);
    assert_eq!(AllenRelation::Meets.inverse(), AllenRelation::MetBy);
    assert_eq!(
        AllenRelation::Overlaps.inverse(),
        AllenRelation::OverlappedBy
    );
    assert_eq!(AllenRelation::During.inverse(), AllenRelation::Contains);
    assert_eq!(AllenRelation::Starts.inverse(), AllenRelation::StartedBy);
    assert_eq!(AllenRelation::Finishes.inverse(), AllenRelation::FinishedBy);
    assert_eq!(AllenRelation::Equals.inverse(), AllenRelation::Equals);
}

/// Test temporal constraint network
#[test]
fn test_temporal_constraint_network_basic() {
    use crate::temporal::{AllenRelation, TemporalConstraintNetwork, TimeInterval};

    let mut tcn = TemporalConstraintNetwork::new();
    let morning = TimeInterval::new(8.0, 12.0).expect("morning interval should be valid");
    let afternoon = TimeInterval::new(13.0, 17.0).expect("afternoon interval should be valid");

    tcn.add_interval("morning".to_string(), morning);
    tcn.add_interval("afternoon".to_string(), afternoon);
    tcn.add_constraint(
        "morning".to_string(),
        "afternoon".to_string(),
        AllenRelation::Before,
    );

    assert!(
        tcn.is_consistent()
            .expect("consistency check should succeed"),
        "morning before afternoon should be consistent"
    );
}

// =========================================================================
// NEGATION TESTS
// =========================================================================

/// Test NAF (negation-as-failure) basic semantics
#[test]
fn test_negation_as_failure_basic() {
    use crate::negation::{NafAtom, NafRule, StratificationConfig, StratifiedReasoner};

    let config = StratificationConfig::default();
    let mut reasoner = StratifiedReasoner::new(config);

    // Rule: notFather(X, Y) :- person(X), person(Y), NOT parent(X, Y)
    let naf_rule = NafRule::new(
        "not_father_rule".to_string(),
        vec![
            NafAtom::Positive(RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("person".to_string()),
            }),
            NafAtom::Positive(RuleAtom::Triple {
                subject: Term::Variable("Y".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("person".to_string()),
            }),
            NafAtom::Negated(RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Variable("Y".to_string()),
            }),
        ],
        vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("notParentOf".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
    );

    assert!(naf_rule.has_negation(), "rule should have negation");

    reasoner.add_rule(naf_rule);
    assert!(
        reasoner.is_stratified(),
        "rules should be properly stratified"
    );
}

/// Test NAF rule structure validation
#[test]
fn test_naf_rule_predicate_analysis() {
    use crate::negation::{NafAtom, NafRule};

    let naf_rule = NafRule::new(
        "test_rule".to_string(),
        vec![
            NafAtom::Positive(RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("works".to_string()),
                object: Term::Variable("Y".to_string()),
            }),
            NafAtom::Negated(RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("fired".to_string()),
                object: Term::Variable("Y".to_string()),
            }),
        ],
        vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("employed".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
    );

    let pos_preds = naf_rule.positive_body_predicates();
    let neg_preds = naf_rule.negated_body_predicates();
    let head_preds = naf_rule.head_predicates();

    assert!(
        pos_preds.contains("works"),
        "works should be positive predicate"
    );
    assert!(
        neg_preds.contains("fired"),
        "fired should be negated predicate"
    );
    assert!(
        head_preds.contains("employed"),
        "employed should be head predicate"
    );
    assert!(naf_rule.has_negation(), "rule should have negation");
}

/// Test stratification analysis with no cycles
#[test]
fn test_stratification_acyclic_rules() {
    use crate::negation::{NafAtom, NafRule, StratificationAnalyzer, StratificationConfig};

    let analyzer = StratificationAnalyzer::new(StratificationConfig::default());

    let rules = vec![
        NafRule::new(
            "base_rule".to_string(),
            vec![NafAtom::Positive(RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("entity".to_string()),
                object: Term::Constant("true".to_string()),
            })],
            vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("exists".to_string()),
                object: Term::Constant("true".to_string()),
            }],
        ),
        NafRule::new(
            "derived_rule".to_string(),
            vec![
                NafAtom::Positive(RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("exists".to_string()),
                    object: Term::Constant("true".to_string()),
                }),
                NafAtom::Negated(RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("deleted".to_string()),
                    object: Term::Constant("true".to_string()),
                }),
            ],
            vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("active".to_string()),
                object: Term::Constant("true".to_string()),
            }],
        ),
    ];

    let result = analyzer.analyze(&rules);
    assert!(result.is_stratified, "acyclic rules should be stratified");
}

/// Test StratifiedReasoner closed-world assumption
#[test]
fn test_stratified_reasoner_cwa() {
    use crate::negation::{StratificationConfig, StratifiedReasoner};

    let config = StratificationConfig::default();
    let mut reasoner = StratifiedReasoner::new(config);

    // Add a fact that alice is employed
    reasoner.add_fact("(alice,employed,company)");

    // alice is known to be employed
    assert!(
        reasoner.is_known("(alice,employed,company)"),
        "alice should be known as employed"
    );
    // bob is NOT known (CWA: assume false)
    assert!(
        reasoner.is_not_known("(bob,employed,company)"),
        "bob should not be known"
    );
}

// =========================================================================
// PROBABILISTIC RULES TESTS
// =========================================================================

/// Test Bayesian Network creation and basic structure
#[test]
fn test_bayesian_network_basic() {
    use crate::probabilistic::BayesianNetwork;

    let mut bn = BayesianNetwork::new();

    bn.add_variable(
        "Rain".to_string(),
        vec!["true".to_string(), "false".to_string()],
    );
    bn.add_variable(
        "WetGrass".to_string(),
        vec!["true".to_string(), "false".to_string()],
    );

    assert_eq!(bn.variable_count(), 2, "should have 2 variables");
    assert!(bn.has_variable("Rain"), "should have Rain variable");
    assert!(bn.has_variable("WetGrass"), "should have WetGrass variable");
    assert!(
        !bn.has_variable("NonExistent"),
        "should not have NonExistent variable"
    );
}

/// Test Bayesian Network edge operations
#[test]
fn test_bayesian_network_edges() {
    use crate::probabilistic::BayesianNetwork;

    let mut bn = BayesianNetwork::new();
    bn.add_variable(
        "Rain".to_string(),
        vec!["true".to_string(), "false".to_string()],
    );
    bn.add_variable(
        "WetGrass".to_string(),
        vec!["true".to_string(), "false".to_string()],
    );

    bn.add_edge("Rain".to_string(), "WetGrass".to_string())
        .expect("add_edge should succeed");

    assert!(
        bn.has_edge("Rain", "WetGrass"),
        "should have Rain -> WetGrass edge"
    );
    assert!(
        !bn.has_edge("WetGrass", "Rain"),
        "should not have reverse edge"
    );
}

/// Test Bayesian Network prior probability
#[test]
fn test_bayesian_network_prior() {
    use crate::probabilistic::BayesianNetwork;

    let mut bn = BayesianNetwork::new();
    bn.add_variable(
        "Rain".to_string(),
        vec!["true".to_string(), "false".to_string()],
    );

    bn.set_prior("Rain".to_string(), vec![0.3, 0.7])
        .expect("set_prior should succeed");

    let prior = bn.get_prior("Rain", "true");
    assert!((prior - 0.3).abs() < 1e-10, "Rain=true prior should be 0.3");
}

/// Test Markov Logic Network basic operations
#[test]
fn test_markov_logic_network_basic() {
    use crate::probabilistic::MarkovLogicNetwork;

    let mut mln = MarkovLogicNetwork::new();

    mln.add_predicate("friends".to_string(), 2);
    mln.add_predicate("smokes".to_string(), 1);

    assert!(
        mln.has_predicate("friends"),
        "should have 'friends' predicate"
    );
    assert!(
        mln.has_predicate("smokes"),
        "should have 'smokes' predicate"
    );
    assert!(
        !mln.has_predicate("unknown"),
        "should not have 'unknown' predicate"
    );

    mln.add_formula("friends(X,Y) => smokes(X) <=> smokes(Y)".to_string(), 1.5);

    assert_eq!(mln.formula_count(), 1, "should have 1 formula");
}

// =========================================================================
// EDGE CASES TESTS
// =========================================================================

/// Test empty knowledge base forward chaining
#[test]
fn test_empty_knowledge_base_forward_chain() {
    let mut engine = RuleEngine::new();

    let empty_facts: Vec<RuleAtom> = vec![];
    let results = engine
        .forward_chain(&empty_facts)
        .expect("empty forward chain should succeed");
    assert!(
        results.is_empty(),
        "empty facts should produce empty results"
    );
}

/// Test empty knowledge base backward chaining
#[test]
fn test_empty_knowledge_base_backward_chain() {
    let mut engine = RuleEngine::new();

    let goal = RuleAtom::Triple {
        subject: Term::Constant("a".to_string()),
        predicate: Term::Constant("b".to_string()),
        object: Term::Constant("c".to_string()),
    };

    let proved = engine
        .backward_chain(&goal)
        .expect("backward chain should return result");
    assert!(!proved, "empty knowledge base should not prove any goal");
}

/// Test rule engine with no rules but facts
#[test]
fn test_no_rules_with_facts() {
    let mut engine = RuleEngine::new();

    let facts = vec![RuleAtom::Triple {
        subject: Term::Constant("a".to_string()),
        predicate: Term::Constant("b".to_string()),
        object: Term::Constant("c".to_string()),
    }];

    engine.add_facts(facts.clone());
    let all_facts = engine.get_facts();

    // With no rules, only the input facts should exist
    assert_eq!(all_facts.len(), 1, "should only have 1 fact with no rules");
}

/// Test rules that produce no new facts
#[test]
fn test_rules_produce_no_new_facts() {
    let mut engine = RuleEngine::new();

    // Rule requires predicates that don't exist in the facts
    engine.add_rule(Rule {
        name: "unmatching_rule".to_string(),
        body: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("nonexistent_pred".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("new_pred".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
    });

    let facts = vec![RuleAtom::Triple {
        subject: Term::Constant("a".to_string()),
        predicate: Term::Constant("different_pred".to_string()),
        object: Term::Constant("b".to_string()),
    }];

    let results = engine
        .forward_chain(&facts)
        .expect("should succeed even when rule doesn't match");
    // Should have only the original fact — rule doesn't match
    assert_eq!(
        results.len(),
        1,
        "should only return original fact when rule doesn't match"
    );
}

/// Test that clear() clears all facts
#[test]
fn test_engine_clear() {
    let mut engine = RuleEngine::new();

    engine.add_rule(Rule {
        name: "test_rule".to_string(),
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

    let facts = vec![RuleAtom::Triple {
        subject: Term::Constant("x".to_string()),
        predicate: Term::Constant("a".to_string()),
        object: Term::Constant("y".to_string()),
    }];

    engine.forward_chain(&facts).expect("should succeed");
    let pre_clear = engine.get_facts().len();
    assert!(pre_clear > 0, "should have facts before clear");

    engine.clear();
    let post_clear = engine.get_facts().len();
    assert_eq!(post_clear, 0, "should have no facts after clear");
}

/// Test add_fact (single) and get_facts
#[test]
fn test_add_single_fact() {
    let mut engine = RuleEngine::new();

    let fact = RuleAtom::Triple {
        subject: Term::Constant("s".to_string()),
        predicate: Term::Constant("p".to_string()),
        object: Term::Constant("o".to_string()),
    };

    engine.add_fact(fact);
    let facts = engine.get_facts();

    assert_eq!(facts.len(), 1, "should have exactly one fact");
}

/// Test backward chaining fails on missing intermediate fact
#[test]
fn test_backward_chain_missing_intermediate() {
    let mut engine = RuleEngine::new();

    // Rule: result(X) :- intermediate(X)
    engine.add_rule(Rule {
        name: "result_rule".to_string(),
        body: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("intermediate".to_string()),
            object: Term::Constant("true".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("result".to_string()),
            object: Term::Constant("true".to_string()),
        }],
    });

    // No intermediate fact for alice — goal should fail
    let goal = RuleAtom::Triple {
        subject: Term::Constant("alice".to_string()),
        predicate: Term::Constant("result".to_string()),
        object: Term::Constant("true".to_string()),
    };

    let proved = engine
        .backward_chain(&goal)
        .expect("backward chain should return result");
    assert!(
        !proved,
        "goal should fail when intermediate fact is missing"
    );
}

// =========================================================================
// PERFORMANCE TESTS
// =========================================================================

/// Test performance with 500+ facts and multiple rules
#[test]
fn test_performance_500_facts_multiple_rules() {
    let mut engine = RuleEngine::new();

    engine.add_rules(vec![
        Rule {
            name: "type_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("category".to_string()),
                object: Term::Constant("item".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("classified".to_string()),
                object: Term::Constant("true".to_string()),
            }],
        },
        Rule {
            name: "status_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("classified".to_string()),
                object: Term::Constant("true".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("processed".to_string()),
                object: Term::Constant("true".to_string()),
            }],
        },
    ]);

    let mut facts = Vec::new();
    for i in 0..500 {
        facts.push(RuleAtom::Triple {
            subject: Term::Constant(format!("item_{i}")),
            predicate: Term::Constant("category".to_string()),
            object: Term::Constant("item".to_string()),
        });
    }

    let start = std::time::Instant::now();
    let results = engine
        .forward_chain(&facts)
        .expect("500-fact inference should succeed");
    let duration = start.elapsed();

    assert!(duration.as_secs() < 5, "should complete within 5 seconds");
    // Each item generates: classified + processed = 2 derived + 1 original = 3 facts each
    assert!(
        results.len() >= 1500,
        "should produce at least 1500 facts (500 * 3)"
    );
}

/// Test performance with RETE for large fact sets
#[test]
fn test_rete_performance_large_fact_set() {
    let mut engine = RuleEngine::new();

    engine.add_rule(Rule {
        name: "rete_perf_rule".to_string(),
        body: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("has_data".to_string()),
            object: Term::Variable("V".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("processed".to_string()),
            object: Term::Variable("V".to_string()),
        }],
    });

    let mut facts = Vec::new();
    for i in 0..200 {
        facts.push(RuleAtom::Triple {
            subject: Term::Constant(format!("node_{i}")),
            predicate: Term::Constant("has_data".to_string()),
            object: Term::Constant(format!("value_{i}")),
        });
    }

    let start = std::time::Instant::now();
    let results = engine
        .rete_forward_chain(facts)
        .expect("RETE performance test should succeed");
    let duration = start.elapsed();

    assert!(
        duration.as_secs() < 5,
        "RETE should complete within 5 seconds"
    );
    assert!(
        results.len() >= 400,
        "RETE should produce at least 400 facts"
    );
}

// =========================================================================
// CACHE MANAGEMENT TESTS
// =========================================================================

/// Test cache enable/disable operations
#[test]
fn test_cache_enable_disable() {
    let mut engine = RuleEngine::new();

    assert!(
        engine.is_cache_enabled(),
        "cache should be enabled by default"
    );

    engine.disable_cache();
    assert!(
        !engine.is_cache_enabled(),
        "cache should be disabled after disable_cache"
    );

    engine.enable_cache();
    assert!(
        engine.is_cache_enabled(),
        "cache should be enabled after enable_cache"
    );
}

/// Test cache statistics
#[test]
fn test_cache_statistics() {
    let engine = RuleEngine::new();

    let stats = engine.get_cache_statistics();
    assert!(
        stats.is_some(),
        "cache statistics should be available when cache is enabled"
    );
}

/// Test cache warm-up with facts
#[test]
fn test_cache_warmup() {
    let mut engine = RuleEngine::new();

    engine.add_rule(Rule {
        name: "warm_rule".to_string(),
        body: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("base".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("derived".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
    });

    let common_facts = vec![RuleAtom::Triple {
        subject: Term::Constant("warm_subject".to_string()),
        predicate: Term::Constant("base".to_string()),
        object: Term::Constant("warm_object".to_string()),
    }];

    // warm_cache should not panic or return an error
    engine.warm_cache(&common_facts);
}

/// Test setting cache with None disables it
#[test]
fn test_set_cache_none_disables() {
    let mut engine = RuleEngine::new();

    engine.set_cache(None);
    assert!(
        !engine.is_cache_enabled(),
        "setting cache to None should disable it"
    );
    assert!(engine.get_cache().is_none(), "get_cache should return None");
}

// =========================================================================
// TERM AND RULE ATOM TESTS
// =========================================================================

/// Test Term equality
#[test]
fn test_term_equality() {
    let const1 = Term::Constant("abc".to_string());
    let const2 = Term::Constant("abc".to_string());
    let const3 = Term::Constant("xyz".to_string());
    let var1 = Term::Variable("X".to_string());
    let var2 = Term::Variable("X".to_string());
    let lit1 = Term::Literal("123".to_string());

    assert_eq!(const1, const2, "same constants should be equal");
    assert_ne!(const1, const3, "different constants should not be equal");
    assert_eq!(var1, var2, "same variables should be equal");
    assert_ne!(const1, var1, "constant and variable should not be equal");
    assert_ne!(const1, lit1, "constant and literal should not be equal");
}

/// Test RuleAtom equality
#[test]
fn test_rule_atom_equality() {
    let atom1 = RuleAtom::Triple {
        subject: Term::Constant("s".to_string()),
        predicate: Term::Constant("p".to_string()),
        object: Term::Constant("o".to_string()),
    };
    let atom2 = RuleAtom::Triple {
        subject: Term::Constant("s".to_string()),
        predicate: Term::Constant("p".to_string()),
        object: Term::Constant("o".to_string()),
    };
    let atom3 = RuleAtom::Triple {
        subject: Term::Constant("s".to_string()),
        predicate: Term::Constant("p".to_string()),
        object: Term::Constant("different".to_string()),
    };

    assert_eq!(atom1, atom2, "identical atoms should be equal");
    assert_ne!(atom1, atom3, "different atoms should not be equal");
}

/// Test builtin RuleAtom
#[test]
fn test_builtin_rule_atom() {
    let builtin = RuleAtom::Builtin {
        name: "equal".to_string(),
        args: vec![
            Term::Constant("a".to_string()),
            Term::Constant("a".to_string()),
        ],
    };

    let builtin2 = RuleAtom::Builtin {
        name: "equal".to_string(),
        args: vec![
            Term::Constant("a".to_string()),
            Term::Constant("a".to_string()),
        ],
    };

    assert_eq!(builtin, builtin2, "identical builtins should be equal");
}

/// Test NotEqual, GreaterThan, LessThan atoms
#[test]
fn test_comparison_atoms() {
    let not_eq = RuleAtom::NotEqual {
        left: Term::Constant("a".to_string()),
        right: Term::Constant("b".to_string()),
    };
    let gt = RuleAtom::GreaterThan {
        left: Term::Constant("10".to_string()),
        right: Term::Constant("5".to_string()),
    };
    let lt = RuleAtom::LessThan {
        left: Term::Constant("3".to_string()),
        right: Term::Constant("7".to_string()),
    };

    // Test they are distinct variants
    assert_ne!(not_eq, gt, "NotEqual and GreaterThan should differ");
    assert_ne!(gt, lt, "GreaterThan and LessThan should differ");
}

// =========================================================================
// RULE ENGINE DEFAULT AND HELPER TESTS
// =========================================================================

/// Test RuleEngine::default()
#[test]
fn test_rule_engine_default() {
    let engine = RuleEngine::default();
    let facts = engine.get_facts();
    assert!(facts.is_empty(), "default engine should have no facts");
}

/// Test add_rules with empty vector
#[test]
fn test_add_rules_empty_vector() {
    let mut engine = RuleEngine::new();
    engine.add_rules(vec![]);
    let facts = engine.get_facts();
    assert!(
        facts.is_empty(),
        "adding empty rules should not affect facts"
    );
}

/// Test backward chaining with direct fact match
#[test]
fn test_backward_chain_direct_match() {
    let mut engine = RuleEngine::new();

    let fact = RuleAtom::Triple {
        subject: Term::Constant("alice".to_string()),
        predicate: Term::Constant("knows".to_string()),
        object: Term::Constant("bob".to_string()),
    };

    engine.add_fact(fact.clone());

    let proved = engine
        .backward_chain(&fact)
        .expect("backward chain should succeed");
    assert!(proved, "directly added fact should be provable");
}

/// Test Rule serialization round-trip
#[test]
fn test_rule_serialization() {
    let rule = Rule {
        name: "test_serialize".to_string(),
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
    };

    let serialized = serde_json::to_string(&rule).expect("serialization should succeed");
    let deserialized: Rule =
        serde_json::from_str(&serialized).expect("deserialization should succeed");

    assert_eq!(
        rule.name, deserialized.name,
        "rule name should survive serialization"
    );
    assert_eq!(
        rule.body.len(),
        deserialized.body.len(),
        "rule body length should survive serialization"
    );
}

/// Test max depth configuration for backward chaining
#[test]
fn test_backward_chain_max_depth_config() {
    let mut engine = RuleEngine::new();

    // Set low max depth to prevent deep recursion
    engine.set_backward_chain_max_depth(5);

    // Add a rule that would cause deep recursion
    engine.add_rule(Rule {
        name: "deep_rule".to_string(),
        body: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("child".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("descendant".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
    });

    let fact = RuleAtom::Triple {
        subject: Term::Constant("a".to_string()),
        predicate: Term::Constant("child".to_string()),
        object: Term::Constant("b".to_string()),
    };

    engine.add_fact(fact);

    // This should succeed without stack overflow due to max depth limit
    let goal = RuleAtom::Triple {
        subject: Term::Constant("a".to_string()),
        predicate: Term::Constant("descendant".to_string()),
        object: Term::Constant("b".to_string()),
    };

    let proved = engine
        .backward_chain(&goal)
        .expect("backward chain with depth limit should succeed");
    assert!(proved, "simple goal should be provable within depth limit");
}

/// Test multi-variable rule with NotEqual constraint
#[test]
fn test_rule_not_equal_constraint() {
    let mut engine = RuleEngine::new();

    // Rule: different(X, Y) :- entity(X), entity(Y), X != Y
    engine.add_rule(Rule {
        name: "different_rule".to_string(),
        body: vec![
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("entity".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Variable("Y".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("entity".to_string()),
            },
            RuleAtom::NotEqual {
                left: Term::Variable("X".to_string()),
                right: Term::Variable("Y".to_string()),
            },
        ],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("differentFrom".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
    });

    let facts = vec![
        RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("entity".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("bob".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("entity".to_string()),
        },
    ];

    let results = engine
        .forward_chain(&facts)
        .expect("not-equal constraint rule should succeed");

    // Should derive differentFrom for alice-bob and bob-alice
    let alice_diff_bob = results.iter().any(|fact| {
        matches!(fact, RuleAtom::Triple {
            subject: Term::Constant(s),
            predicate: Term::Constant(p),
            object: Term::Constant(o)
        } if s == "alice" && p == "differentFrom" && o == "bob")
    });

    assert!(alice_diff_bob, "alice differentFrom bob should be derived");
}

/// Test GreaterThan constraint in rules
#[test]
fn test_rule_greater_than_constraint() {
    let mut engine = RuleEngine::new();

    // Rule: highValue(X) :- hasValue(X, V), V > 100
    engine.add_rule(Rule {
        name: "high_value_rule".to_string(),
        body: vec![
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("value".to_string()),
                object: Term::Variable("V".to_string()),
            },
            RuleAtom::GreaterThan {
                left: Term::Variable("V".to_string()),
                right: Term::Constant("100".to_string()),
            },
        ],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("category".to_string()),
            object: Term::Constant("highValue".to_string()),
        }],
    });

    let facts = vec![
        RuleAtom::Triple {
            subject: Term::Constant("item1".to_string()),
            predicate: Term::Constant("value".to_string()),
            object: Term::Constant("150".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("item2".to_string()),
            predicate: Term::Constant("value".to_string()),
            object: Term::Constant("50".to_string()),
        },
    ];

    let results = engine
        .forward_chain(&facts)
        .expect("greater-than constraint should work");

    let item1_high = results.iter().any(|fact| {
        matches!(fact, RuleAtom::Triple {
            subject: Term::Constant(s),
            predicate: Term::Constant(p),
            object: Term::Constant(o)
        } if s == "item1" && p == "category" && o == "highValue")
    });

    let item2_high = results.iter().any(|fact| {
        matches!(fact, RuleAtom::Triple {
            subject: Term::Constant(s),
            predicate: Term::Constant(p),
            object: Term::Constant(o)
        } if s == "item2" && p == "category" && o == "highValue")
    });

    assert!(item1_high, "item1 (value 150) should be high value");
    assert!(!item2_high, "item2 (value 50) should NOT be high value");
}

/// Test multi-body rule with two join conditions
#[test]
fn test_two_body_join_rule() {
    let mut engine = RuleEngine::new();

    // Rule: qualified(X, Y) :- eligible(X), available(Y)
    engine.add_rule(Rule {
        name: "qualified_rule".to_string(),
        body: vec![
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("eligible".to_string()),
                object: Term::Constant("true".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Variable("Y".to_string()),
                predicate: Term::Constant("available".to_string()),
                object: Term::Constant("true".to_string()),
            },
        ],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("qualifiedFor".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
    });

    let facts = vec![
        RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant("eligible".to_string()),
            object: Term::Constant("true".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("role_engineer".to_string()),
            predicate: Term::Constant("available".to_string()),
            object: Term::Constant("true".to_string()),
        },
    ];

    let results = engine
        .forward_chain(&facts)
        .expect("join rule should succeed");

    assert!(
        results.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "alice" && p == "qualifiedFor" && o == "role_engineer")
        }),
        "alice should be qualifiedFor role_engineer via join"
    );
}

/// Test RDFS context hierarchy operations
#[test]
fn test_rdfs_context_hierarchy() {
    use crate::rdfs::RdfsContext;

    let mut context = RdfsContext::default();

    context.add_subclass_relation("Mammal", "Animal");
    context.add_subclass_relation("Dog", "Mammal");

    // Transitive closure: Dog is subclass of Animal
    assert!(
        context.is_subclass_of("Dog", "Mammal"),
        "Dog should be subclass of Mammal"
    );
    assert!(
        context.is_subclass_of("Dog", "Animal"),
        "Dog should be transitive subclass of Animal"
    );
    assert!(
        !context.is_subclass_of("Animal", "Dog"),
        "Animal should NOT be subclass of Dog"
    );
}

/// Test OWL context consistency check
#[test]
fn test_owl_context_consistency() {
    use crate::owl::OwlContext;

    let mut context = OwlContext::default();

    // Add equivalence and disjointness — should detect inconsistency
    context.add_equivalent_classes("ClassA", "ClassB");
    context.add_disjoint_classes("ClassA", "ClassB");

    let is_consistent = context.check_consistency();
    assert!(
        !is_consistent,
        "equivalent and disjoint classes should be inconsistent"
    );
}

/// Test OWL context with same/different individuals
#[test]
fn test_owl_context_same_different_individuals() {
    use crate::owl::OwlContext;

    let mut context = OwlContext::default();

    // Add that ind1 is same as ind2, and ind1 is different from ind2 — contradiction
    context.add_same_individuals("ind1", "ind2");
    context.add_different_individuals("ind1", "ind2");

    let is_consistent = context.check_consistency();
    assert!(
        !is_consistent,
        "same and different assertions for same individuals should be inconsistent"
    );
}

/// Test RDFS builder pattern
#[test]
fn test_rdfs_builder_pattern() {
    use crate::rdfs::{RdfsProfile, RdfsReasoner, RdfsRule};

    let reasoner = RdfsReasoner::builder()
        .with_profile(RdfsProfile::Minimal)
        .enable_rule(RdfsRule::Rdfs4a)
        .build();

    assert!(
        reasoner.is_rule_enabled(RdfsRule::Rdfs4a),
        "Rdfs4a should be enabled after builder configuration"
    );
    assert!(
        reasoner.is_rule_enabled(RdfsRule::Rdfs9),
        "Rdfs9 should be enabled in Minimal profile"
    );
}

/// Test RDFS profile minimal vs full
#[test]
fn test_rdfs_profile_minimal_vs_full() {
    use crate::rdfs::{RdfsProfile, RdfsReasoner, RdfsRule};

    let minimal_reasoner = RdfsReasoner::with_profile(RdfsProfile::Minimal);
    let full_reasoner = RdfsReasoner::with_profile(RdfsProfile::Full);

    // Rdfs1 should be disabled in Minimal but enabled in Full
    assert!(
        !minimal_reasoner.is_rule_enabled(RdfsRule::Rdfs1),
        "Rdfs1 should NOT be enabled in Minimal profile"
    );
    assert!(
        full_reasoner.is_rule_enabled(RdfsRule::Rdfs1),
        "Rdfs1 should be enabled in Full profile"
    );

    // Rdfs9 should be enabled in both
    assert!(
        minimal_reasoner.is_rule_enabled(RdfsRule::Rdfs9),
        "Rdfs9 should be enabled in Minimal profile"
    );
    assert!(
        full_reasoner.is_rule_enabled(RdfsRule::Rdfs9),
        "Rdfs9 should be enabled in Full profile"
    );
}

/// Test forward chainer derive_new_facts
#[test]
fn test_forward_chainer_derive_new_facts() {
    use crate::forward::ForwardChainer;

    let mut chainer = ForwardChainer::new();

    chainer.add_rule(Rule {
        name: "derive_rule".to_string(),
        body: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("source".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("derived".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
    });

    chainer.add_fact(RuleAtom::Triple {
        subject: Term::Constant("node".to_string()),
        predicate: Term::Constant("source".to_string()),
        object: Term::Constant("target".to_string()),
    });

    let new_facts = chainer
        .derive_new_facts()
        .expect("derive_new_facts should succeed");

    assert!(
        new_facts.iter().any(|fact| {
            matches!(fact, RuleAtom::Triple {
                subject: Term::Constant(s),
                predicate: Term::Constant(p),
                object: Term::Constant(o)
            } if s == "node" && p == "derived" && o == "target")
        }),
        "should derive new fact via derive_new_facts"
    );
}

/// Test backward chainer query
#[test]
fn test_backward_chainer_query() {
    use crate::backward::BackwardChainer;

    let mut chainer = BackwardChainer::new();

    chainer.add_fact(RuleAtom::Triple {
        subject: Term::Constant("alice".to_string()),
        predicate: Term::Constant("knows".to_string()),
        object: Term::Constant("bob".to_string()),
    });
    chainer.add_fact(RuleAtom::Triple {
        subject: Term::Constant("alice".to_string()),
        predicate: Term::Constant("knows".to_string()),
        object: Term::Constant("carol".to_string()),
    });

    let pattern = RuleAtom::Triple {
        subject: Term::Constant("alice".to_string()),
        predicate: Term::Constant("knows".to_string()),
        object: Term::Variable("Z".to_string()),
    };

    let results = chainer.query(&pattern).expect("query should succeed");
    assert_eq!(results.len(), 2, "alice should know 2 people");
}

/// Test Function term in rules
#[test]
fn test_function_term_in_rules() {
    let func_term = Term::Function {
        name: "concat".to_string(),
        args: vec![
            Term::Constant("hello".to_string()),
            Term::Constant("world".to_string()),
        ],
    };

    let func_term2 = Term::Function {
        name: "concat".to_string(),
        args: vec![
            Term::Constant("hello".to_string()),
            Term::Constant("world".to_string()),
        ],
    };

    assert_eq!(
        func_term, func_term2,
        "identical function terms should be equal"
    );

    let func_term3 = Term::Function {
        name: "concat".to_string(),
        args: vec![
            Term::Constant("foo".to_string()),
            Term::Constant("bar".to_string()),
        ],
    };

    assert_ne!(
        func_term, func_term3,
        "different function terms should not be equal"
    );
}
