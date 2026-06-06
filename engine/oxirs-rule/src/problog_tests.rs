//! Probabilistic Datalog (ProbLog) — tests
//!
//! All `#[cfg(test)]` blocks for the ProbLog engine.

#[cfg(test)]
mod tests {
    use crate::{Rule, RuleAtom, Term};
    use anyhow::Result;

    use crate::problog_solver::ProbLogEngine;
    use crate::problog_types::{EvaluationStrategy, ProbabilisticFact, ProbabilisticRule};

    fn create_triple(subject: &str, predicate: &str, object: &str) -> RuleAtom {
        RuleAtom::Triple {
            subject: Term::Constant(subject.to_string()),
            predicate: Term::Constant(predicate.to_string()),
            object: Term::Constant(object.to_string()),
        }
    }

    #[test]
    fn test_probabilistic_fact_creation() -> Result<()> {
        let fact = ProbabilisticFact::new(0.8, create_triple("john", "parent", "mary"))?;
        assert_eq!(fact.probability, 0.8);
        Ok(())
    }

    #[test]
    fn test_invalid_probability() {
        let result = ProbabilisticFact::new(1.5, create_triple("john", "parent", "mary"));
        assert!(result.is_err());
    }

    #[test]
    fn test_query_deterministic_fact() -> Result<()> {
        let mut engine = ProbLogEngine::new();
        let fact = create_triple("john", "parent", "mary");

        engine.add_fact(fact.clone());

        let prob = engine.query_probability(&fact)?;
        assert_eq!(prob, 1.0);

        Ok(())
    }

    #[test]
    fn test_query_probabilistic_fact() -> Result<()> {
        let mut engine = ProbLogEngine::new();
        let fact = create_triple("john", "parent", "mary");

        engine.add_probabilistic_fact(ProbabilisticFact::new(0.8, fact.clone())?);

        let prob = engine.query_probability(&fact)?;
        assert_eq!(prob, 0.8);

        Ok(())
    }

    #[test]
    fn test_query_unknown_fact() -> Result<()> {
        let mut engine = ProbLogEngine::new();
        let fact = create_triple("john", "parent", "mary");

        let prob = engine.query_probability(&fact)?;
        assert_eq!(prob, 0.0);

        Ok(())
    }

    #[test]
    fn test_rule_derivation() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.8,
            create_triple("john", "parent", "mary"),
        )?);

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor".to_string(),
            body: vec![create_triple("john", "parent", "mary")],
            head: vec![create_triple("john", "ancestor", "mary")],
        }));

        let ancestor_fact = create_triple("john", "ancestor", "mary");
        let prob = engine.query_probability(&ancestor_fact)?;

        assert!((prob - 0.8).abs() < 0.001);

        Ok(())
    }

    #[test]
    fn test_query_caching() -> Result<()> {
        let mut engine = ProbLogEngine::new();
        let fact = create_triple("john", "parent", "mary");

        engine.add_probabilistic_fact(ProbabilisticFact::new(0.7, fact.clone())?);

        engine.query_probability(&fact)?;
        assert_eq!(engine.stats.cache_misses, 1);
        assert_eq!(engine.stats.cache_hits, 0);

        engine.query_probability(&fact)?;
        assert_eq!(engine.stats.cache_hits, 1);

        Ok(())
    }

    #[test]
    fn test_sampling() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        engine.add_fact(create_triple("john", "person", "true"));

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.5,
            create_triple("john", "tall", "true"),
        )?);

        let mut tall_count = 0;
        let samples = 1000;

        for _ in 0..samples {
            let world = engine.sample();
            if world.contains(&create_triple("john", "tall", "true")) {
                tall_count += 1;
            }
            assert!(world.contains(&create_triple("john", "person", "true")));
        }

        let proportion = tall_count as f64 / samples as f64;
        assert!((proportion - 0.5).abs() < 0.1);

        Ok(())
    }

    #[test]
    fn test_explanation_tree() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.9,
            create_triple("john", "parent", "mary"),
        )?);

        let tree = engine.explain(&create_triple("john", "parent", "mary"))?;

        assert!(tree.is_some());
        let tree = tree.ok_or_else(|| anyhow::anyhow!("expected Some value"))?;
        assert_eq!(tree.probability, 0.9);
        assert!(tree.premises.is_empty());

        Ok(())
    }

    #[test]
    fn test_probabilistic_rule() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        engine.add_fact(create_triple("john", "parent", "mary"));

        engine.add_rule(ProbabilisticRule::probabilistic(
            0.9,
            Rule {
                name: "ancestor".to_string(),
                body: vec![create_triple("john", "parent", "mary")],
                head: vec![create_triple("john", "ancestor", "mary")],
            },
        )?);

        let prob = engine.query_probability(&create_triple("john", "ancestor", "mary"))?;

        assert!((prob - 0.9).abs() < 0.001);

        Ok(())
    }

    #[test]
    fn test_variable_unification_simple() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.8,
            create_triple("john", "parent", "mary"),
        )?);

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor_rule".to_string(),
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
        }));

        let prob = engine.query_probability(&create_triple("john", "ancestor", "mary"))?;

        assert!((prob - 0.8).abs() < 0.001, "Expected 0.8, got {}", prob);

        Ok(())
    }

    #[test]
    fn test_variable_unification_multiple_facts() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.9,
            create_triple("john", "parent", "mary"),
        )?);

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.7,
            create_triple("mary", "parent", "bob"),
        )?);

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor".to_string(),
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
        }));

        let prob1 = engine.query_probability(&create_triple("john", "ancestor", "mary"))?;
        assert!((prob1 - 0.9).abs() < 0.001, "Expected 0.9, got {}", prob1);

        let prob2 = engine.query_probability(&create_triple("mary", "ancestor", "bob"))?;
        assert!((prob2 - 0.7).abs() < 0.001, "Expected 0.7, got {}", prob2);

        Ok(())
    }

    #[test]
    fn test_variable_unification_transitive() -> Result<()> {
        let mut engine = ProbLogEngine::new().with_strategy(EvaluationStrategy::Auto);

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.9,
            create_triple("john", "parent", "mary"),
        )?);

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.8,
            create_triple("mary", "parent", "bob"),
        )?);

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor_base".to_string(),
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
        }));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor_trans".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("parent".to_string()),
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
        }));

        let prob = engine.query_probability(&create_triple("john", "ancestor", "bob"))?;

        assert!((prob - 0.72).abs() < 0.001, "Expected 0.72, got {}", prob);

        assert!(
            engine.stats.fixpoint_iterations > 0,
            "Should have used fixpoint iteration for recursive rules"
        );

        Ok(())
    }

    #[test]
    fn test_variable_unification_with_probabilistic_rule() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        engine.add_fact(create_triple("john", "parent", "mary"));

        engine.add_rule(ProbabilisticRule::probabilistic(
            0.95,
            Rule {
                name: "related_rule".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("parent".to_string()),
                    object: Term::Variable("Y".to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("related".to_string()),
                    object: Term::Variable("Y".to_string()),
                }],
            },
        )?);

        let prob = engine.query_probability(&create_triple("john", "related", "mary"))?;

        assert!((prob - 0.95).abs() < 0.001, "Expected 0.95, got {}", prob);

        Ok(())
    }

    #[test]
    fn test_cycle_detection() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        engine.add_fact(create_triple("a", "edge", "b"));
        engine.add_fact(create_triple("b", "edge", "c"));
        engine.add_fact(create_triple("c", "edge", "a"));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "path_transitive".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("edge".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("path".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        }));

        let prob = engine.query_probability(&create_triple("a", "path", "a"))?;

        assert_eq!(
            prob, 0.0,
            "Cycle detection should return 0.0 for cyclic queries"
        );

        Ok(())
    }

    #[test]
    fn test_unification_failure() -> Result<()> {
        let mut engine = ProbLogEngine::new();

        engine.add_fact(create_triple("john", "parent", "mary"));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "specific_rule".to_string(),
            body: vec![create_triple("john", "parent", "bob")],
            head: vec![create_triple("john", "ancestor", "bob")],
        }));

        let prob = engine.query_probability(&create_triple("john", "ancestor", "bob"))?;

        assert!(
            prob.abs() < 0.001,
            "Expected 0.0, got {} - unification should fail",
            prob
        );

        Ok(())
    }

    #[test]
    fn test_fixpoint_transitive_closure() -> Result<()> {
        let mut engine = ProbLogEngine::new().with_strategy(EvaluationStrategy::BottomUp);

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.9,
            create_triple("john", "parent", "mary"),
        )?);

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.8,
            create_triple("mary", "parent", "bob"),
        )?);

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor_base".to_string(),
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
        }));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor_trans".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("parent".to_string()),
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
        }));

        let prob1 = engine.query_probability(&create_triple("john", "ancestor", "mary"))?;
        assert!((prob1 - 0.9).abs() < 0.001, "Expected 0.9, got {}", prob1);

        let prob2 = engine.query_probability(&create_triple("mary", "ancestor", "bob"))?;
        assert!((prob2 - 0.8).abs() < 0.001, "Expected 0.8, got {}", prob2);

        let prob3 = engine.query_probability(&create_triple("john", "ancestor", "bob"))?;
        assert!(
            (prob3 - 0.72).abs() < 0.001,
            "Expected 0.72 (transitive), got {}",
            prob3
        );

        assert!(
            engine.stats.fixpoint_iterations > 0,
            "Should have used fixpoint iteration"
        );
        assert!(
            engine.stats.materialized_facts_count >= 5,
            "Should have materialized at least 5 facts (2 parent + 3 ancestor)"
        );

        Ok(())
    }

    #[test]
    fn test_fixpoint_cyclic_graph() -> Result<()> {
        let mut engine = ProbLogEngine::new().with_strategy(EvaluationStrategy::BottomUp);

        engine.add_fact(create_triple("a", "edge", "b"));
        engine.add_fact(create_triple("b", "edge", "c"));
        engine.add_fact(create_triple("c", "edge", "a"));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "path_base".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("edge".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "path_transitive".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("edge".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("path".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        }));

        let prob = engine.query_probability(&create_triple("a", "path", "a"))?;

        assert_eq!(
            prob, 1.0,
            "Fixpoint iteration should correctly compute cyclic path (got {})",
            prob
        );

        let prob_ab = engine.query_probability(&create_triple("a", "path", "b"))?;
        let prob_bc = engine.query_probability(&create_triple("b", "path", "c"))?;
        let prob_ca = engine.query_probability(&create_triple("c", "path", "a"))?;

        assert_eq!(prob_ab, 1.0, "path(a,b) should be 1.0");
        assert_eq!(prob_bc, 1.0, "path(b,c) should be 1.0");
        assert_eq!(prob_ca, 1.0, "path(c,a) should be 1.0");

        Ok(())
    }

    #[test]
    fn test_fixpoint_auto_strategy() -> Result<()> {
        let mut engine = ProbLogEngine::new().with_strategy(EvaluationStrategy::Auto);

        engine.add_fact(create_triple("john", "parent", "mary"));
        engine.add_fact(create_triple("mary", "parent", "bob"));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor_base".to_string(),
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
        }));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "ancestor_trans".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("parent".to_string()),
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
        }));

        let prob = engine.query_probability(&create_triple("john", "ancestor", "bob"))?;
        assert_eq!(prob, 1.0, "Auto strategy should compute transitive closure");

        assert!(
            engine.stats.fixpoint_iterations > 0,
            "Auto strategy should have used fixpoint iteration for recursive rules"
        );

        Ok(())
    }

    #[test]
    fn test_fixpoint_probabilistic_combination() -> Result<()> {
        let mut engine = ProbLogEngine::new().with_strategy(EvaluationStrategy::BottomUp);

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.6,
            create_triple("a", "edge", "b"),
        )?);

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.7,
            create_triple("a", "edge", "c"),
        )?);

        engine.add_probabilistic_fact(ProbabilisticFact::new(
            0.8,
            create_triple("c", "edge", "b"),
        )?);

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "connected_direct".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("edge".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("connected".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "connected_indirect".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("edge".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Z".to_string()),
                    predicate: Term::Constant("edge".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("connected".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        let prob = engine.query_probability(&create_triple("a", "connected", "b"))?;

        let expected = 0.6 + 0.56 - (0.6 * 0.56); // 0.824
        assert!(
            (prob - expected).abs() < 0.001,
            "Expected {} (disjunctive combination), got {}",
            expected,
            prob
        );

        Ok(())
    }

    #[test]
    fn test_fixpoint_max_iterations() -> Result<()> {
        let mut engine = ProbLogEngine::new()
            .with_strategy(EvaluationStrategy::BottomUp)
            .with_max_fixpoint_iterations(2);

        engine.add_fact(create_triple("a", "edge", "b"));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "path".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("edge".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        let result = engine.query_probability(&create_triple("a", "path", "b"));
        assert!(result.is_ok(), "Should succeed with low iteration limit");

        Ok(())
    }

    #[test]
    fn test_materialization_invalidation() -> Result<()> {
        let mut engine = ProbLogEngine::new().with_strategy(EvaluationStrategy::BottomUp);

        engine.add_fact(create_triple("a", "edge", "b"));
        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "path".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("edge".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        let prob1 = engine.query_probability(&create_triple("a", "path", "b"))?;
        assert_eq!(prob1, 1.0);
        let iter1 = engine.stats.fixpoint_iterations;

        engine.add_fact(create_triple("b", "edge", "c"));

        let prob2 = engine.query_probability(&create_triple("b", "path", "c"))?;
        assert_eq!(prob2, 1.0);

        assert!(
            engine.stats.fixpoint_iterations >= iter1,
            "Should have re-materialized after adding fact"
        );

        Ok(())
    }

    #[test]
    fn test_fixpoint_statistics() -> Result<()> {
        let mut engine = ProbLogEngine::new().with_strategy(EvaluationStrategy::BottomUp);

        engine.add_fact(create_triple("a", "edge", "b"));
        engine.add_fact(create_triple("b", "edge", "c"));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "path_base".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("edge".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }));

        engine.add_rule(ProbabilisticRule::deterministic(Rule {
            name: "path_trans".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("path".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("edge".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        }));

        let _prob = engine.query_probability(&create_triple("a", "path", "c"))?;

        assert!(
            engine.stats.fixpoint_iterations > 0,
            "Should have recorded fixpoint iterations"
        );
        assert!(
            engine.stats.materialized_facts_count > 0,
            "Should have recorded materialized facts count"
        );
        assert!(engine.stats.queries > 0, "Should have recorded query count");

        Ok(())
    }
}
