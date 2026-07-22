//! SPARQL Compliance Tests — aggregation (`aggregation_tests` +
//! `aggregate_extended_tests`).
//!
//! Split out of `sparql_compliance.rs` to keep each integration test source
//! file below the 2000-line refactoring policy. Shared fixtures live in
//! `tests/common/mod.rs`.

mod common;

use common::{create_test_dataset, MockDataset};
use oxirs_arq::algebra::Term;
use oxirs_arq::{
    Aggregate, Algebra, Expression, GroupCondition, Literal, QueryExecutor, TriplePattern, Variable,
};
use oxirs_core::model::NamedNode;

#[cfg(test)]
mod aggregation_tests {
    use super::*;

    #[test]
    fn test_count_aggregation() {
        let dataset = create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Query: SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }
        let algebra = Algebra::Group {
            pattern: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("s").unwrap()),
                predicate: Term::Variable(Variable::new("p").unwrap()),
                object: Term::Variable(Variable::new("o").unwrap()),
            }])),
            variables: vec![], // No grouping variables = single group
            aggregates: vec![(
                Variable::new("count").unwrap(),
                Aggregate::Count {
                    distinct: false,
                    expr: None, // COUNT(*)
                },
            )],
        };

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        println!("Solution length: {}", solution.len());
        for (i, binding) in solution.iter().enumerate() {
            println!("Binding {i}: {binding:?}");
        }

        assert_eq!(solution.len(), 1);
        let count = solution[0].get(&Variable::new("count").unwrap()).unwrap();
        match count {
            Term::Literal(lit) => assert_eq!(lit.value, "4"), // 4 triples in test dataset
            _ => panic!("Expected literal"),
        }
    }

    #[test]
    fn test_group_by_with_count() {
        let dataset = create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Add more data for grouping
        let mut dataset = dataset;
        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked("http://example.org/alice")),
            Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/knows")),
            Term::Iri(NamedNode::new_unchecked("http://example.org/bob")),
        );

        // Query: SELECT ?person (COUNT(*) as ?count) WHERE { ?person ?p ?o } GROUP BY ?person
        let algebra = Algebra::Group {
            pattern: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("person").unwrap()),
                predicate: Term::Variable(Variable::new("p").unwrap()),
                object: Term::Variable(Variable::new("o").unwrap()),
            }])),
            variables: vec![GroupCondition {
                expr: Expression::Variable(Variable::new("person").unwrap()),
                alias: None,
            }],
            aggregates: vec![(
                Variable::new("count").unwrap(),
                Aggregate::Count {
                    distinct: false,
                    expr: None,
                },
            )],
        };

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        // Should have 2 groups (Alice and Bob)
        assert_eq!(solution.len(), 2);

        // Alice should have count of 3 (name, age, knows)
        let alice_binding = solution
            .iter()
            .find(|binding| {
                binding
                    .get(&Variable::new("person").unwrap())
                    .and_then(|term| match term {
                        Term::Iri(iri) => Some(iri.as_str()),
                        _ => None,
                    })
                    .is_some_and(|iri| iri.contains("alice"))
            })
            .unwrap();

        let alice_count = alice_binding.get(&Variable::new("count").unwrap()).unwrap();
        match alice_count {
            Term::Literal(lit) => assert_eq!(lit.value, "3"),
            _ => panic!("Expected literal"),
        }
    }
}

// ============================================================
// aggregate_extended_tests — 7 tests for SUM, MIN, MAX, AVG,
// SAMPLE, GROUP_CONCAT, and DISTINCT COUNT aggregates
// ============================================================
#[cfg(test)]
mod aggregate_extended_tests {
    use super::*;

    fn build_dataset_with_scores() -> MockDataset {
        let mut dataset = MockDataset::new();
        let xsd_int = NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer");

        // Alice has score 80
        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked("http://example.org/alice")),
            Term::Iri(NamedNode::new_unchecked("http://example.org/score")),
            Term::Literal(Literal {
                value: "80".to_string(),
                language: None,
                datatype: Some(xsd_int.clone()),
            }),
        );
        // Bob has score 60
        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked("http://example.org/bob")),
            Term::Iri(NamedNode::new_unchecked("http://example.org/score")),
            Term::Literal(Literal {
                value: "60".to_string(),
                language: None,
                datatype: Some(xsd_int.clone()),
            }),
        );
        // Charlie has score 70
        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked("http://example.org/charlie")),
            Term::Iri(NamedNode::new_unchecked("http://example.org/score")),
            Term::Literal(Literal {
                value: "70".to_string(),
                language: None,
                datatype: Some(xsd_int),
            }),
        );
        dataset
    }

    fn score_bgp() -> Algebra {
        Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(Variable::new("person").expect("valid variable")),
            predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/score")),
            object: Term::Variable(Variable::new("score").expect("valid variable")),
        }])
    }

    #[test]
    fn test_sum_aggregation() {
        let dataset = build_dataset_with_scores();
        let mut executor = QueryExecutor::new();

        // SELECT (SUM(?score) AS ?total) WHERE { ?person ex:score ?score }
        let algebra = Algebra::Group {
            pattern: Box::new(score_bgp()),
            variables: vec![],
            aggregates: vec![(
                Variable::new("total").expect("valid variable"),
                Aggregate::Sum {
                    distinct: false,
                    expr: Expression::Variable(Variable::new("score").expect("valid variable")),
                },
            )],
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(solution.len(), 1, "SUM aggregate returns exactly one row");

        let total = solution[0]
            .get(&Variable::new("total").expect("valid variable"))
            .expect("total must be bound");
        match total {
            Term::Literal(lit) => {
                let total_val: f64 = lit.value.parse().expect("sum value must be numeric");
                // 80 + 60 + 70 = 210
                assert!(
                    (total_val - 210.0).abs() < 0.001,
                    "SUM must be 210, got {total_val}"
                );
            }
            _ => panic!("Expected literal for SUM result"),
        }
    }

    /// Regression: SUM over xsd:integer operands must keep xsd:integer typing
    /// (not be widened to xsd:decimal) on the live group-by execution path.
    #[test]
    fn regression_sum_integer_typing_live_path() {
        let dataset = build_dataset_with_scores();
        let mut executor = QueryExecutor::new();
        let algebra = Algebra::Group {
            pattern: Box::new(score_bgp()),
            variables: vec![],
            aggregates: vec![(
                Variable::new("total").expect("valid variable"),
                Aggregate::Sum {
                    distinct: false,
                    expr: Expression::Variable(Variable::new("score").expect("valid variable")),
                },
            )],
        };
        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        let total = solution[0]
            .get(&Variable::new("total").expect("valid variable"))
            .expect("total must be bound");
        match total {
            Term::Literal(lit) => {
                assert_eq!(lit.value, "210", "exact integer sum");
                assert_eq!(
                    lit.datatype.as_ref().map(|d| d.as_str()),
                    Some("http://www.w3.org/2001/XMLSchema#integer"),
                    "SUM of integers must stay xsd:integer"
                );
            }
            _ => panic!("Expected literal for SUM result"),
        }
    }

    #[test]
    fn test_min_aggregation() {
        let dataset = build_dataset_with_scores();
        let mut executor = QueryExecutor::new();

        // SELECT (MIN(?score) AS ?min_score)
        let algebra = Algebra::Group {
            pattern: Box::new(score_bgp()),
            variables: vec![],
            aggregates: vec![(
                Variable::new("min_score").expect("valid variable"),
                Aggregate::Min {
                    distinct: false,
                    expr: Expression::Variable(Variable::new("score").expect("valid variable")),
                },
            )],
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(solution.len(), 1);

        let min_val = solution[0]
            .get(&Variable::new("min_score").expect("valid variable"))
            .expect("min_score must be bound");
        match min_val {
            Term::Literal(lit) => {
                let n: f64 = lit.value.parse().expect("min value must be numeric");
                assert!((n - 60.0).abs() < 0.001, "MIN must be 60, got {n}");
            }
            _ => panic!("Expected literal for MIN result"),
        }
    }

    #[test]
    fn test_max_aggregation() {
        let dataset = build_dataset_with_scores();
        let mut executor = QueryExecutor::new();

        // SELECT (MAX(?score) AS ?max_score)
        let algebra = Algebra::Group {
            pattern: Box::new(score_bgp()),
            variables: vec![],
            aggregates: vec![(
                Variable::new("max_score").expect("valid variable"),
                Aggregate::Max {
                    distinct: false,
                    expr: Expression::Variable(Variable::new("score").expect("valid variable")),
                },
            )],
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(solution.len(), 1);

        let max_val = solution[0]
            .get(&Variable::new("max_score").expect("valid variable"))
            .expect("max_score must be bound");
        match max_val {
            Term::Literal(lit) => {
                let n: f64 = lit.value.parse().expect("max value must be numeric");
                assert!((n - 80.0).abs() < 0.001, "MAX must be 80, got {n}");
            }
            _ => panic!("Expected literal for MAX result"),
        }
    }

    #[test]
    fn test_avg_aggregation() {
        let dataset = build_dataset_with_scores();
        let mut executor = QueryExecutor::new();

        // SELECT (AVG(?score) AS ?avg_score)
        let algebra = Algebra::Group {
            pattern: Box::new(score_bgp()),
            variables: vec![],
            aggregates: vec![(
                Variable::new("avg_score").expect("valid variable"),
                Aggregate::Avg {
                    distinct: false,
                    expr: Expression::Variable(Variable::new("score").expect("valid variable")),
                },
            )],
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(solution.len(), 1);

        let avg_val = solution[0]
            .get(&Variable::new("avg_score").expect("valid variable"))
            .expect("avg_score must be bound");
        match avg_val {
            Term::Literal(lit) => {
                let n: f64 = lit.value.parse().expect("avg value must be numeric");
                // (80 + 60 + 70) / 3 = 70.0
                assert!((n - 70.0).abs() < 0.001, "AVG must be 70.0, got {n}");
            }
            _ => panic!("Expected literal for AVG result"),
        }
    }

    #[test]
    fn test_sample_aggregation() {
        let dataset = build_dataset_with_scores();
        let mut executor = QueryExecutor::new();

        // SELECT (SAMPLE(?score) AS ?any_score) — returns one of the scores
        let algebra = Algebra::Group {
            pattern: Box::new(score_bgp()),
            variables: vec![],
            aggregates: vec![(
                Variable::new("any_score").expect("valid variable"),
                Aggregate::Sample {
                    distinct: false,
                    expr: Expression::Variable(Variable::new("score").expect("valid variable")),
                },
            )],
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(solution.len(), 1, "SAMPLE returns exactly one row");

        let sample_val = solution[0]
            .get(&Variable::new("any_score").expect("valid variable"))
            .expect("any_score must be bound");
        match sample_val {
            Term::Literal(lit) => {
                let n: f64 = lit.value.parse().expect("sample must be numeric");
                assert!(
                    [60.0_f64, 70.0, 80.0]
                        .iter()
                        .any(|&v| (n - v).abs() < 0.001),
                    "SAMPLE value must be one of the scores (60, 70, or 80), got {n}"
                );
            }
            _ => panic!("Expected literal for SAMPLE result"),
        }
    }

    #[test]
    fn test_group_concat_aggregation() {
        let dataset = build_dataset_with_scores();
        let mut executor = QueryExecutor::new();

        // SELECT (GROUP_CONCAT(?person; SEPARATOR=", ") AS ?persons)
        // Concatenates all person IRIs
        let algebra = Algebra::Group {
            pattern: Box::new(score_bgp()),
            variables: vec![],
            aggregates: vec![(
                Variable::new("persons").expect("valid variable"),
                Aggregate::GroupConcat {
                    distinct: false,
                    expr: Expression::Variable(Variable::new("person").expect("valid variable")),
                    separator: Some(", ".to_string()),
                },
            )],
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(solution.len(), 1, "GROUP_CONCAT returns exactly one row");

        let persons = solution[0]
            .get(&Variable::new("persons").expect("valid variable"))
            .expect("persons must be bound");
        match persons {
            Term::Literal(lit) => {
                // Should be a concatenation of three IRIs
                assert!(
                    !lit.value.is_empty(),
                    "GROUP_CONCAT result must not be empty"
                );
            }
            _ => panic!("Expected literal for GROUP_CONCAT result"),
        }
    }

    #[test]
    fn test_count_distinct_aggregation() {
        // Dataset where same name appears multiple times — COUNT DISTINCT should deduplicate
        let mut dataset = MockDataset::new();
        let alice_iri = Term::Iri(NamedNode::new_unchecked("http://example.org/alice"));
        let name_pred = Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name"));
        let alice_name = Term::Literal(Literal {
            value: "Alice".to_string(),
            language: None,
            datatype: None,
        });

        // Alice appears twice (same name but different triples would be unusual, but for testing)
        dataset.add_triple(alice_iri.clone(), name_pred.clone(), alice_name.clone());
        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked("http://example.org/alice2")),
            name_pred,
            alice_name,
        );

        let mut executor = QueryExecutor::new();

        // SELECT (COUNT(DISTINCT ?name) AS ?distinct_count)
        let algebra = Algebra::Group {
            pattern: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("person").expect("valid variable")),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
                object: Term::Variable(Variable::new("name").expect("valid variable")),
            }])),
            variables: vec![],
            aggregates: vec![(
                Variable::new("distinct_count").expect("valid variable"),
                Aggregate::Count {
                    distinct: true,
                    expr: Some(Expression::Variable(
                        Variable::new("name").expect("valid variable"),
                    )),
                },
            )],
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(solution.len(), 1);

        let distinct_count = solution[0]
            .get(&Variable::new("distinct_count").expect("valid variable"))
            .expect("distinct_count must be bound");
        match distinct_count {
            Term::Literal(lit) => {
                let n: usize = lit.value.parse().expect("count must be integer");
                assert_eq!(n, 1, "COUNT DISTINCT of two identical names must be 1");
            }
            _ => panic!("Expected literal for COUNT DISTINCT"),
        }
    }
}
