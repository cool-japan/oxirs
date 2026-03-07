//! SPARQL Compliance Tests
//!
//! This module provides basic SPARQL compliance testing while we work
//! towards full W3C test suite integration.

use oxirs_arq::algebra::Term;
use oxirs_arq::{
    Algebra, BinaryOperator, Expression, Literal, QueryExecutor, TriplePattern, Variable,
};
use oxirs_core::model::NamedNode;

/// Mock dataset for testing
struct MockDataset {
    triples: Vec<(Term, Term, Term)>,
}

impl MockDataset {
    fn new() -> Self {
        Self {
            triples: Vec::new(),
        }
    }

    fn add_triple(&mut self, s: Term, p: Term, o: Term) {
        self.triples.push((s, p, o));
    }
}

impl oxirs_arq::Dataset for MockDataset {
    fn find_triples(&self, pattern: &TriplePattern) -> anyhow::Result<Vec<(Term, Term, Term)>> {
        let mut results = Vec::new();

        for (s, p, o) in &self.triples {
            let mut matches = true;

            // Check subject
            match &pattern.subject {
                Term::Variable(_) => {} // Variables match anything
                other => {
                    if other != s {
                        matches = false;
                    }
                }
            }

            // Check predicate
            if matches {
                match &pattern.predicate {
                    Term::Variable(_) => {} // Variables match anything
                    other => {
                        if other != p {
                            matches = false;
                        }
                    }
                }
            }

            // Check object
            if matches {
                match &pattern.object {
                    Term::Variable(_) => {} // Variables match anything
                    other => {
                        if other != o {
                            matches = false;
                        }
                    }
                }
            }

            if matches {
                results.push((s.clone(), p.clone(), o.clone()));
            }
        }

        Ok(results)
    }

    fn contains_triple(
        &self,
        subject: &Term,
        predicate: &Term,
        object: &Term,
    ) -> anyhow::Result<bool> {
        Ok(self
            .triples
            .iter()
            .any(|(s, p, o)| s == subject && p == predicate && o == object))
    }

    fn subjects(&self) -> anyhow::Result<Vec<Term>> {
        let mut subjects: Vec<Term> = self.triples.iter().map(|(s, _, _)| s.clone()).collect();
        subjects.sort();
        subjects.dedup();
        Ok(subjects)
    }

    fn predicates(&self) -> anyhow::Result<Vec<Term>> {
        let mut predicates: Vec<Term> = self.triples.iter().map(|(_, p, _)| p.clone()).collect();
        predicates.sort();
        predicates.dedup();
        Ok(predicates)
    }

    fn objects(&self) -> anyhow::Result<Vec<Term>> {
        let mut objects: Vec<Term> = self.triples.iter().map(|(_, _, o)| o.clone()).collect();
        objects.sort();
        objects.dedup();
        Ok(objects)
    }
}

#[cfg(test)]
mod basic_tests {
    use super::*;

    pub fn create_test_dataset() -> MockDataset {
        let mut dataset = MockDataset::new();

        // Add some test data
        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked("http://example.org/alice")),
            Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
            Term::Literal(Literal {
                value: "Alice".to_string(),
                language: None,
                datatype: None,
            }),
        );

        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked("http://example.org/alice")),
            Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/age")),
            Term::Literal(Literal {
                value: "30".to_string(),
                language: None,
                datatype: Some(NamedNode::new_unchecked(
                    "http://www.w3.org/2001/XMLSchema#integer",
                )),
            }),
        );

        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked("http://example.org/bob")),
            Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
            Term::Literal(Literal {
                value: "Bob".to_string(),
                language: None,
                datatype: None,
            }),
        );

        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked("http://example.org/bob")),
            Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/age")),
            Term::Literal(Literal {
                value: "25".to_string(),
                language: None,
                datatype: Some(NamedNode::new_unchecked(
                    "http://www.w3.org/2001/XMLSchema#integer",
                )),
            }),
        );

        dataset
    }

    #[test]
    fn test_basic_bgp() {
        let dataset = create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Query: ?person foaf:name ?name
        let algebra = Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(Variable::new("person").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
            object: Term::Variable(Variable::new("name").unwrap()),
        }]);

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        assert_eq!(solution.len(), 2);

        // Check that we got both Alice and Bob
        let names: Vec<String> = solution
            .iter()
            .filter_map(|binding| {
                binding
                    .get(&Variable::new("name").unwrap())
                    .and_then(|term| match term {
                        Term::Literal(lit) => Some(lit.value.clone()),
                        _ => None,
                    })
            })
            .collect();

        assert!(names.contains(&"Alice".to_string()));
        assert!(names.contains(&"Bob".to_string()));
    }

    #[test]
    fn test_filter_numeric_comparison() {
        let dataset = create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Query: ?person foaf:age ?age . FILTER(?age > 25)
        let algebra = Algebra::Filter {
            pattern: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("person").unwrap()),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/age")),
                object: Term::Variable(Variable::new("age").unwrap()),
            }])),
            condition: Expression::Binary {
                op: BinaryOperator::Greater,
                left: Box::new(Expression::Variable(Variable::new("age").unwrap())),
                right: Box::new(Expression::Literal(Literal {
                    value: "25".to_string(),
                    language: None,
                    datatype: Some(NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                })),
            },
        };

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        // Should only return Alice (age 30)
        assert_eq!(solution.len(), 1);

        let person = solution[0].get(&Variable::new("person").unwrap()).unwrap();
        match person {
            Term::Iri(iri) => assert_eq!(iri.as_str(), "http://example.org/alice"),
            _ => panic!("Expected IRI"),
        }
    }

    #[test]
    fn test_optional_pattern() {
        let dataset = create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Add a person without age
        let mut dataset = dataset;
        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked("http://example.org/charlie")),
            Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
            Term::Literal(Literal {
                value: "Charlie".to_string(),
                language: None,
                datatype: None,
            }),
        );

        // Query: ?person foaf:name ?name . OPTIONAL { ?person foaf:age ?age }
        let algebra = Algebra::LeftJoin {
            left: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("person").unwrap()),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
                object: Term::Variable(Variable::new("name").unwrap()),
            }])),
            right: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("person").unwrap()),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/age")),
                object: Term::Variable(Variable::new("age").unwrap()),
            }])),
            filter: None,
        };

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        // Should return all three people
        assert_eq!(solution.len(), 3);

        // Charlie should have name but no age
        let charlie_binding = solution
            .iter()
            .find(|binding| {
                binding
                    .get(&Variable::new("name").unwrap())
                    .and_then(|term| match term {
                        Term::Literal(lit) => Some(lit.value.as_str()),
                        _ => None,
                    })
                    == Some("Charlie")
            })
            .unwrap();

        assert!(charlie_binding.contains_key(&Variable::new("name").unwrap()));
        assert!(!charlie_binding.contains_key(&Variable::new("age").unwrap()));
    }

    #[test]
    fn test_union_pattern() {
        let dataset = create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Query: { ?s foaf:name "Alice" } UNION { ?s foaf:age "25"^^xsd:integer }
        let algebra = Algebra::Union {
            left: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("s").unwrap()),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
                object: Term::Literal(Literal {
                    value: "Alice".to_string(),
                    language: None,
                    datatype: None,
                }),
            }])),
            right: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("s").unwrap()),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/age")),
                object: Term::Literal(Literal {
                    value: "25".to_string(),
                    language: None,
                    datatype: Some(NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#integer",
                    )),
                }),
            }])),
        };

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        // Should return Alice and Bob (Bob has age 25)
        assert_eq!(solution.len(), 2);

        let subjects: Vec<String> = solution
            .iter()
            .filter_map(|binding| {
                binding
                    .get(&Variable::new("s").unwrap())
                    .and_then(|term| match term {
                        Term::Iri(iri) => Some(iri.as_str().to_string()),
                        _ => None,
                    })
            })
            .collect();

        assert!(subjects.contains(&"http://example.org/alice".to_string()));
        assert!(subjects.contains(&"http://example.org/bob".to_string()));
    }

    #[test]
    fn test_projection() {
        let dataset = create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Query: SELECT ?name WHERE { ?person foaf:name ?name ; foaf:age ?age }
        let algebra = Algebra::Project {
            pattern: Box::new(Algebra::Bgp(vec![
                TriplePattern {
                    subject: Term::Variable(Variable::new("person").unwrap()),
                    predicate: Term::Iri(NamedNode::new_unchecked(
                        "http://xmlns.com/foaf/0.1/name",
                    )),
                    object: Term::Variable(Variable::new("name").unwrap()),
                },
                TriplePattern {
                    subject: Term::Variable(Variable::new("person").unwrap()),
                    predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/age")),
                    object: Term::Variable(Variable::new("age").unwrap()),
                },
            ])),
            variables: vec![Variable::new("name").unwrap()],
        };

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        // Check that only 'name' variable is in results
        for binding in &solution {
            assert!(binding.contains_key(&Variable::new("name").unwrap()));
            assert!(!binding.contains_key(&Variable::new("person").unwrap()));
            assert!(!binding.contains_key(&Variable::new("age").unwrap()));
        }
    }

    #[test]
    fn test_distinct() {
        let dataset = create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Add duplicate data
        let mut dataset = dataset;
        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked("http://example.org/alice2")),
            Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
            Term::Literal(Literal {
                value: "Alice".to_string(),
                language: None,
                datatype: None,
            }),
        );

        // Query: SELECT DISTINCT ?name WHERE { ?person foaf:name ?name }
        let algebra = Algebra::Distinct {
            pattern: Box::new(Algebra::Project {
                pattern: Box::new(Algebra::Bgp(vec![TriplePattern {
                    subject: Term::Variable(Variable::new("person").unwrap()),
                    predicate: Term::Iri(NamedNode::new_unchecked(
                        "http://xmlns.com/foaf/0.1/name",
                    )),
                    object: Term::Variable(Variable::new("name").unwrap()),
                }])),
                variables: vec![Variable::new("name").unwrap()],
            }),
        };

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        // Should have only 2 distinct names (Alice and Bob)
        assert_eq!(solution.len(), 2);
    }

    #[test]
    fn test_order_by() {
        let dataset = create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Query: SELECT ?person ?age WHERE { ?person foaf:age ?age } ORDER BY ?age
        let algebra = Algebra::OrderBy {
            pattern: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("person").unwrap()),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/age")),
                object: Term::Variable(Variable::new("age").unwrap()),
            }])),
            conditions: vec![oxirs_arq::OrderCondition {
                expr: Expression::Variable(Variable::new("age").unwrap()),
                ascending: true,
            }],
        };

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        // Bob (25) should come before Alice (30)
        assert_eq!(solution.len(), 2);

        let first_person = solution[0].get(&Variable::new("person").unwrap()).unwrap();
        match first_person {
            Term::Iri(iri) => assert_eq!(iri.as_str(), "http://example.org/bob"),
            _ => panic!("Expected IRI"),
        }

        let second_person = solution[1].get(&Variable::new("person").unwrap()).unwrap();
        match second_person {
            Term::Iri(iri) => assert_eq!(iri.as_str(), "http://example.org/alice"),
            _ => panic!("Expected IRI"),
        }
    }

    #[test]
    fn test_limit_offset() {
        let dataset = create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Query: SELECT ?person WHERE { ?person foaf:name ?name } LIMIT 1 OFFSET 1
        let algebra = Algebra::Slice {
            pattern: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("person").unwrap()),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
                object: Term::Variable(Variable::new("name").unwrap()),
            }])),
            offset: Some(1),
            limit: Some(1),
        };

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        // Should return exactly 1 result (skipping the first)
        assert_eq!(solution.len(), 1);
    }
}

#[cfg(test)]
mod aggregation_tests {
    use super::*;
    use oxirs_arq::{Aggregate, GroupCondition};

    #[test]
    fn test_count_aggregation() {
        let dataset = super::basic_tests::create_test_dataset();
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
        let dataset = super::basic_tests::create_test_dataset();
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
// filter_advanced_tests — 8 tests covering numeric and boolean
// FILTER combinations not yet in basic_tests
// ============================================================
#[cfg(test)]
mod filter_advanced_tests {
    use super::*;

    fn xsd_integer_lit(value: &str) -> Literal {
        Literal {
            value: value.to_string(),
            language: None,
            datatype: Some(NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#integer",
            )),
        }
    }

    fn age_bgp() -> Algebra {
        Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(Variable::new("person").expect("valid variable")),
            predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/age")),
            object: Term::Variable(Variable::new("age").expect("valid variable")),
        }])
    }

    #[test]
    fn test_filter_less_than() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // FILTER(?age < 30) — should return only Bob (age 25)
        let algebra = Algebra::Filter {
            pattern: Box::new(age_bgp()),
            condition: Expression::Binary {
                op: BinaryOperator::Less,
                left: Box::new(Expression::Variable(
                    Variable::new("age").expect("valid variable"),
                )),
                right: Box::new(Expression::Literal(xsd_integer_lit("30"))),
            },
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            1,
            "Only Bob (age 25) passes FILTER(?age < 30)"
        );

        let person = solution[0]
            .get(&Variable::new("person").expect("valid variable"))
            .expect("person binding must exist");
        match person {
            Term::Iri(iri) => assert!(iri.as_str().contains("bob"), "Should be Bob"),
            _ => panic!("Expected IRI term for person"),
        }
    }

    #[test]
    fn test_filter_less_equal() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // FILTER(?age <= 25) — should return only Bob (age exactly 25)
        let algebra = Algebra::Filter {
            pattern: Box::new(age_bgp()),
            condition: Expression::Binary {
                op: BinaryOperator::LessEqual,
                left: Box::new(Expression::Variable(
                    Variable::new("age").expect("valid variable"),
                )),
                right: Box::new(Expression::Literal(xsd_integer_lit("25"))),
            },
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            1,
            "Only Bob (age 25) passes FILTER(?age <= 25)"
        );
    }

    #[test]
    fn test_filter_greater_equal() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // FILTER(?age >= 30) — should return only Alice (age 30)
        let algebra = Algebra::Filter {
            pattern: Box::new(age_bgp()),
            condition: Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: Box::new(Expression::Variable(
                    Variable::new("age").expect("valid variable"),
                )),
                right: Box::new(Expression::Literal(xsd_integer_lit("30"))),
            },
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            1,
            "Only Alice (age 30) passes FILTER(?age >= 30)"
        );

        let person = solution[0]
            .get(&Variable::new("person").expect("valid variable"))
            .expect("person binding must exist");
        match person {
            Term::Iri(iri) => assert!(iri.as_str().contains("alice"), "Should be Alice"),
            _ => panic!("Expected IRI term for person"),
        }
    }

    #[test]
    fn test_filter_equal_string() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // FILTER(?name = "Alice") — should return Alice's name binding
        let name_bgp = Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(Variable::new("person").expect("valid variable")),
            predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
            object: Term::Variable(Variable::new("name").expect("valid variable")),
        }]);

        let algebra = Algebra::Filter {
            pattern: Box::new(name_bgp),
            condition: Expression::Binary {
                op: BinaryOperator::Equal,
                left: Box::new(Expression::Variable(
                    Variable::new("name").expect("valid variable"),
                )),
                right: Box::new(Expression::Literal(Literal {
                    value: "Alice".to_string(),
                    language: None,
                    datatype: None,
                })),
            },
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            1,
            "Only Alice passes FILTER(?name = \"Alice\")"
        );

        let name = solution[0]
            .get(&Variable::new("name").expect("valid variable"))
            .expect("name binding must exist");
        match name {
            Term::Literal(lit) => assert_eq!(lit.value, "Alice"),
            _ => panic!("Expected literal for name"),
        }
    }

    #[test]
    fn test_filter_not_equal() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // FILTER(?name != "Alice") — should return Bob's binding
        let name_bgp = Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(Variable::new("person").expect("valid variable")),
            predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
            object: Term::Variable(Variable::new("name").expect("valid variable")),
        }]);

        let algebra = Algebra::Filter {
            pattern: Box::new(name_bgp),
            condition: Expression::Binary {
                op: BinaryOperator::NotEqual,
                left: Box::new(Expression::Variable(
                    Variable::new("name").expect("valid variable"),
                )),
                right: Box::new(Expression::Literal(Literal {
                    value: "Alice".to_string(),
                    language: None,
                    datatype: None,
                })),
            },
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            1,
            "Only Bob passes FILTER(?name != \"Alice\")"
        );

        let name = solution[0]
            .get(&Variable::new("name").expect("valid variable"))
            .expect("name binding must exist");
        match name {
            Term::Literal(lit) => assert_eq!(lit.value, "Bob"),
            _ => panic!("Expected literal for name"),
        }
    }

    #[test]
    fn test_filter_and_conjunction() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // FILTER(?age > 20 && ?age < 35) — both Alice (30) and Bob (25) pass
        let algebra = Algebra::Filter {
            pattern: Box::new(age_bgp()),
            condition: Expression::Binary {
                op: BinaryOperator::And,
                left: Box::new(Expression::Binary {
                    op: BinaryOperator::Greater,
                    left: Box::new(Expression::Variable(
                        Variable::new("age").expect("valid variable"),
                    )),
                    right: Box::new(Expression::Literal(xsd_integer_lit("20"))),
                }),
                right: Box::new(Expression::Binary {
                    op: BinaryOperator::Less,
                    left: Box::new(Expression::Variable(
                        Variable::new("age").expect("valid variable"),
                    )),
                    right: Box::new(Expression::Literal(xsd_integer_lit("35"))),
                }),
            },
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            2,
            "Both Alice and Bob pass FILTER(?age > 20 && ?age < 35)"
        );
    }

    #[test]
    fn test_filter_or_disjunction() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // FILTER(?age < 26 || ?age > 29) — Bob (25 < 26) and Alice (30 > 29) both pass
        let algebra = Algebra::Filter {
            pattern: Box::new(age_bgp()),
            condition: Expression::Binary {
                op: BinaryOperator::Or,
                left: Box::new(Expression::Binary {
                    op: BinaryOperator::Less,
                    left: Box::new(Expression::Variable(
                        Variable::new("age").expect("valid variable"),
                    )),
                    right: Box::new(Expression::Literal(xsd_integer_lit("26"))),
                }),
                right: Box::new(Expression::Binary {
                    op: BinaryOperator::Greater,
                    left: Box::new(Expression::Variable(
                        Variable::new("age").expect("valid variable"),
                    )),
                    right: Box::new(Expression::Literal(xsd_integer_lit("29"))),
                }),
            },
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            2,
            "Both pass FILTER(?age < 26 || ?age > 29)"
        );
    }

    #[test]
    fn test_filter_bound_on_optional() {
        // Add Charlie with only name (no age), then FILTER(BOUND(?age)) on OPTIONAL
        // should return only Alice and Bob (who have age bindings)
        let mut dataset = basic_tests::create_test_dataset();
        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked("http://example.org/charlie")),
            Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
            Term::Literal(Literal {
                value: "Charlie".to_string(),
                language: None,
                datatype: None,
            }),
        );

        let mut executor = QueryExecutor::new();

        // OPTIONAL { ?person foaf:age ?age } then FILTER(BOUND(?age))
        let optional_algebra = Algebra::LeftJoin {
            left: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("person").expect("valid variable")),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
                object: Term::Variable(Variable::new("name").expect("valid variable")),
            }])),
            right: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("person").expect("valid variable")),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/age")),
                object: Term::Variable(Variable::new("age").expect("valid variable")),
            }])),
            filter: None,
        };

        let algebra = Algebra::Filter {
            pattern: Box::new(optional_algebra),
            condition: Expression::Bound(Variable::new("age").expect("valid variable")),
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            2,
            "FILTER(BOUND(?age)) keeps only Alice and Bob"
        );

        // Verify that all solutions have the ?age variable bound
        for binding in &solution {
            assert!(
                binding.contains_key(&Variable::new("age").expect("valid variable")),
                "Every remaining solution must have ?age bound"
            );
        }
    }
}

// ============================================================
// limit_offset_tests — 5 tests for LIMIT/OFFSET pagination
// ============================================================
#[cfg(test)]
mod limit_offset_tests {
    use super::*;

    fn name_bgp() -> Algebra {
        Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(Variable::new("person").expect("valid variable")),
            predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
            object: Term::Variable(Variable::new("name").expect("valid variable")),
        }])
    }

    #[test]
    fn test_limit_zero() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        let algebra = Algebra::Slice {
            pattern: Box::new(name_bgp()),
            offset: None,
            limit: Some(0),
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(solution.len(), 0, "LIMIT 0 must return empty results");
    }

    #[test]
    fn test_limit_more_than_results() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Dataset has 2 people, LIMIT 100 should return both
        let algebra = Algebra::Slice {
            pattern: Box::new(name_bgp()),
            offset: None,
            limit: Some(100),
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            2,
            "LIMIT 100 returns all 2 results when fewer exist"
        );
    }

    #[test]
    fn test_offset_zero() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // OFFSET 0 is same as no offset
        let algebra = Algebra::Slice {
            pattern: Box::new(name_bgp()),
            offset: Some(0),
            limit: None,
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(solution.len(), 2, "OFFSET 0 returns all results unchanged");
    }

    #[test]
    fn test_offset_beyond_results() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Dataset has 2 results, OFFSET 100 should return empty
        let algebra = Algebra::Slice {
            pattern: Box::new(name_bgp()),
            offset: Some(100),
            limit: None,
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            0,
            "OFFSET beyond result count returns empty"
        );
    }

    #[test]
    fn test_limit_and_offset_combined() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // LIMIT 1 OFFSET 0 — returns exactly the first result
        let algebra = Algebra::Slice {
            pattern: Box::new(name_bgp()),
            offset: Some(0),
            limit: Some(1),
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            1,
            "LIMIT 1 OFFSET 0 returns exactly 1 result"
        );
    }
}

// ============================================================
// order_by_tests — 5 tests for ORDER BY variations
// ============================================================
#[cfg(test)]
mod order_by_tests {
    use super::*;

    fn age_bgp() -> Algebra {
        Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(Variable::new("person").expect("valid variable")),
            predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/age")),
            object: Term::Variable(Variable::new("age").expect("valid variable")),
        }])
    }

    fn name_bgp() -> Algebra {
        Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(Variable::new("person").expect("valid variable")),
            predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
            object: Term::Variable(Variable::new("name").expect("valid variable")),
        }])
    }

    #[test]
    fn test_order_by_descending() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // ORDER BY DESC(?age) — Alice (30) should come before Bob (25)
        let algebra = Algebra::OrderBy {
            pattern: Box::new(age_bgp()),
            conditions: vec![oxirs_arq::OrderCondition {
                expr: Expression::Variable(Variable::new("age").expect("valid variable")),
                ascending: false,
            }],
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(solution.len(), 2);

        let first_person = solution[0]
            .get(&Variable::new("person").expect("valid variable"))
            .expect("person must be bound");
        match first_person {
            Term::Iri(iri) => assert!(
                iri.as_str().contains("alice"),
                "Alice (age 30) must be first in DESC order"
            ),
            _ => panic!("Expected IRI"),
        }
    }

    #[test]
    fn test_order_by_string_ascending() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // ORDER BY ASC(?name) — "Alice" < "Bob" alphabetically
        let algebra = Algebra::OrderBy {
            pattern: Box::new(name_bgp()),
            conditions: vec![oxirs_arq::OrderCondition {
                expr: Expression::Variable(Variable::new("name").expect("valid variable")),
                ascending: true,
            }],
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(solution.len(), 2);

        let first_name = solution[0]
            .get(&Variable::new("name").expect("valid variable"))
            .expect("name must be bound");
        match first_name {
            Term::Literal(lit) => assert_eq!(lit.value, "Alice", "Alice sorts before Bob"),
            _ => panic!("Expected literal for name"),
        }
    }

    #[test]
    fn test_order_by_with_limit() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // ORDER BY ?age LIMIT 1 — returns only the row with smallest age (Bob, 25)
        let algebra = Algebra::Slice {
            pattern: Box::new(Algebra::OrderBy {
                pattern: Box::new(age_bgp()),
                conditions: vec![oxirs_arq::OrderCondition {
                    expr: Expression::Variable(Variable::new("age").expect("valid variable")),
                    ascending: true,
                }],
            }),
            offset: None,
            limit: Some(1),
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            1,
            "ORDER BY + LIMIT 1 returns exactly one result"
        );

        let person = solution[0]
            .get(&Variable::new("person").expect("valid variable"))
            .expect("person must be bound");
        match person {
            Term::Iri(iri) => assert!(
                iri.as_str().contains("bob"),
                "Bob (age 25) is first in ASC order"
            ),
            _ => panic!("Expected IRI"),
        }
    }

    #[test]
    fn test_order_by_multiple_conditions() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Multi-key sort: ORDER BY ?person ASC, ?age DESC
        // Both bindings returned; test just verifies stable multi-key sort executes without error
        let algebra = Algebra::OrderBy {
            pattern: Box::new(age_bgp()),
            conditions: vec![
                oxirs_arq::OrderCondition {
                    expr: Expression::Variable(Variable::new("person").expect("valid variable")),
                    ascending: true,
                },
                oxirs_arq::OrderCondition {
                    expr: Expression::Variable(Variable::new("age").expect("valid variable")),
                    ascending: false,
                },
            ],
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(solution.len(), 2, "Multi-key ORDER BY returns all results");
    }

    #[test]
    fn test_order_then_slice_chained() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Slice(OrderBy(age_bgp, desc), offset=1, limit=1) — second-highest age
        // Alice=30, Bob=25 → DESC order: Alice first, Bob second → skip 1, take 1 = Bob
        let algebra = Algebra::Slice {
            pattern: Box::new(Algebra::OrderBy {
                pattern: Box::new(age_bgp()),
                conditions: vec![oxirs_arq::OrderCondition {
                    expr: Expression::Variable(Variable::new("age").expect("valid variable")),
                    ascending: false,
                }],
            }),
            offset: Some(1),
            limit: Some(1),
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            1,
            "OFFSET 1 LIMIT 1 after DESC sort returns second row"
        );

        let person = solution[0]
            .get(&Variable::new("person").expect("valid variable"))
            .expect("person must be bound");
        match person {
            Term::Iri(iri) => assert!(
                iri.as_str().contains("bob"),
                "Bob (age 25) is second in DESC order"
            ),
            _ => panic!("Expected IRI"),
        }
    }
}

// ============================================================
// union_tests — 4 tests for UNION semantics
// ============================================================
#[cfg(test)]
mod union_tests {
    use super::*;

    #[test]
    fn test_union_both_sides_match() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // { ?s foaf:name "Alice" } UNION { ?s foaf:name "Bob" }
        // Both sides match one result each → 2 total
        let alice_name = Term::Literal(Literal {
            value: "Alice".to_string(),
            language: None,
            datatype: None,
        });
        let bob_name = Term::Literal(Literal {
            value: "Bob".to_string(),
            language: None,
            datatype: None,
        });

        let algebra = Algebra::Union {
            left: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("s").expect("valid variable")),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
                object: alice_name,
            }])),
            right: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("s").expect("valid variable")),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
                object: bob_name,
            }])),
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            2,
            "UNION of two single-match patterns returns 2 results"
        );
    }

    #[test]
    fn test_union_left_empty() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Left side matches nothing (nonexistent name), right side matches Bob
        let nonexistent = Term::Literal(Literal {
            value: "NonExistent".to_string(),
            language: None,
            datatype: None,
        });
        let bob_name = Term::Literal(Literal {
            value: "Bob".to_string(),
            language: None,
            datatype: None,
        });

        let algebra = Algebra::Union {
            left: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("s").expect("valid variable")),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
                object: nonexistent,
            }])),
            right: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("s").expect("valid variable")),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
                object: bob_name,
            }])),
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            1,
            "UNION with empty left returns only right results"
        );
    }

    #[test]
    fn test_union_right_empty() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Left matches Alice, right matches nothing
        let alice_name = Term::Literal(Literal {
            value: "Alice".to_string(),
            language: None,
            datatype: None,
        });
        let nonexistent = Term::Literal(Literal {
            value: "NonExistent".to_string(),
            language: None,
            datatype: None,
        });

        let algebra = Algebra::Union {
            left: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("s").expect("valid variable")),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
                object: alice_name,
            }])),
            right: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("s").expect("valid variable")),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
                object: nonexistent,
            }])),
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        assert_eq!(
            solution.len(),
            1,
            "UNION with empty right returns only left results"
        );
    }

    #[test]
    fn test_union_overlapping_without_distinct() {
        let dataset = basic_tests::create_test_dataset();
        let mut executor = QueryExecutor::new();

        // Both sides match all names — without DISTINCT, duplicates are allowed
        let all_names_bgp = || {
            Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("s").expect("valid variable")),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
                object: Term::Variable(Variable::new("name").expect("valid variable")),
            }])
        };

        let algebra = Algebra::Union {
            left: Box::new(all_names_bgp()),
            right: Box::new(all_names_bgp()),
        };

        let (solution, _stats) = executor
            .execute(&algebra, &dataset)
            .expect("execution must succeed");
        // 2 triples matched by each side, UNION concatenates → 4 total (duplicates allowed)
        assert_eq!(
            solution.len(),
            4,
            "UNION without DISTINCT allows duplicate solutions"
        );
    }
}

// ============================================================
// algebra_construction_tests — 10 tests verifying that all key
// Algebra variants can be constructed and their structure is
// correct (covering Extend, Minus, Values, Having, Graph, etc.)
// ============================================================
#[cfg(test)]
mod algebra_construction_tests {
    use super::*;
    use std::collections::HashMap;

    fn make_bgp(var_name: &str) -> Algebra {
        Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(Variable::new(var_name).expect("valid variable")),
            predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
            object: Term::Variable(Variable::new("name").expect("valid variable")),
        }])
    }

    #[test]
    fn test_algebra_extend_construction() {
        // BIND(?name AS ?label) — verifies Extend variant can be constructed
        let var_name = Variable::new("name").expect("valid variable");
        let var_label = Variable::new("label").expect("valid variable");

        let algebra = Algebra::Extend {
            pattern: Box::new(make_bgp("s")),
            variable: var_label.clone(),
            expr: Expression::Variable(var_name),
        };

        match algebra {
            Algebra::Extend { variable, .. } => {
                assert_eq!(variable, var_label, "Extend variable is the bind target");
            }
            _ => panic!("Expected Extend algebra"),
        }
    }

    #[test]
    fn test_algebra_minus_construction() {
        // { ?s foaf:name ?name } MINUS { ?s foaf:name "Alice" }
        let left = make_bgp("s");
        let right = Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(Variable::new("s").expect("valid variable")),
            predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
            object: Term::Literal(Literal {
                value: "Alice".to_string(),
                language: None,
                datatype: None,
            }),
        }]);

        let algebra = Algebra::Minus {
            left: Box::new(left),
            right: Box::new(right),
        };

        assert!(
            matches!(algebra, Algebra::Minus { .. }),
            "Minus variant constructed correctly"
        );
    }

    #[test]
    fn test_algebra_values_construction() {
        // VALUES ?x { <http://example.org/alice> <http://example.org/bob> }
        let var_x = Variable::new("x").expect("valid variable");

        let mut binding1: HashMap<Variable, Term> = HashMap::new();
        binding1.insert(
            var_x.clone(),
            Term::Iri(NamedNode::new_unchecked("http://example.org/alice")),
        );

        let mut binding2: HashMap<Variable, Term> = HashMap::new();
        binding2.insert(
            var_x.clone(),
            Term::Iri(NamedNode::new_unchecked("http://example.org/bob")),
        );

        let algebra = Algebra::Values {
            variables: vec![var_x.clone()],
            bindings: vec![binding1, binding2],
        };

        match &algebra {
            Algebra::Values {
                variables,
                bindings,
            } => {
                assert_eq!(variables.len(), 1, "Values has one variable");
                assert_eq!(variables[0], var_x);
                assert_eq!(bindings.len(), 2, "Values has two binding rows");
            }
            _ => panic!("Expected Values algebra"),
        }
    }

    #[test]
    fn test_algebra_having_construction() {
        // GROUP BY ?person HAVING(?count > 1)
        let count_var = Variable::new("count").expect("valid variable");
        let person_var = Variable::new("person").expect("valid variable");

        let group_algebra = Algebra::Group {
            pattern: Box::new(make_bgp("person")),
            variables: vec![oxirs_arq::GroupCondition {
                expr: Expression::Variable(person_var),
                alias: None,
            }],
            aggregates: vec![(
                count_var.clone(),
                oxirs_arq::Aggregate::Count {
                    distinct: false,
                    expr: None,
                },
            )],
        };

        let having_condition = Expression::Binary {
            op: BinaryOperator::Greater,
            left: Box::new(Expression::Variable(count_var)),
            right: Box::new(Expression::Literal(Literal {
                value: "1".to_string(),
                language: None,
                datatype: Some(NamedNode::new_unchecked(
                    "http://www.w3.org/2001/XMLSchema#integer",
                )),
            })),
        };

        let algebra = Algebra::Having {
            pattern: Box::new(group_algebra),
            condition: having_condition,
        };

        assert!(
            matches!(algebra, Algebra::Having { .. }),
            "Having variant constructed correctly"
        );
    }

    #[test]
    fn test_algebra_graph_construction() {
        // GRAPH <http://example.org/g> { ?s ?p ?o }
        let graph_name = Term::Iri(NamedNode::new_unchecked("http://example.org/g"));
        let inner = Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(Variable::new("s").expect("valid variable")),
            predicate: Term::Variable(Variable::new("p").expect("valid variable")),
            object: Term::Variable(Variable::new("o").expect("valid variable")),
        }]);

        let algebra = Algebra::Graph {
            graph: graph_name.clone(),
            pattern: Box::new(inner),
        };

        match algebra {
            Algebra::Graph { graph, .. } => {
                assert_eq!(
                    graph, graph_name,
                    "Graph name preserved in Graph algebra variant"
                );
            }
            _ => panic!("Expected Graph algebra"),
        }
    }

    #[test]
    fn test_algebra_reduced_construction() {
        // SELECT REDUCED ?name WHERE { ?s foaf:name ?name }
        let inner = Algebra::Project {
            pattern: Box::new(make_bgp("s")),
            variables: vec![Variable::new("name").expect("valid variable")],
        };

        let algebra = Algebra::Reduced {
            pattern: Box::new(inner),
        };

        assert!(
            matches!(algebra, Algebra::Reduced { .. }),
            "Reduced variant constructed correctly"
        );
    }

    #[test]
    fn test_algebra_table_and_zero_variants() {
        // Table and Zero are constant algebra expressions with no patterns
        let table = Algebra::Table;
        let zero = Algebra::Zero;
        let empty = Algebra::Empty;

        assert!(matches!(table, Algebra::Table), "Table variant constructed");
        assert!(matches!(zero, Algebra::Zero), "Zero variant constructed");
        assert!(matches!(empty, Algebra::Empty), "Empty variant constructed");
    }

    #[test]
    fn test_property_path_sequence_display() {
        use oxirs_arq::algebra::PropertyPath;

        let p1 = PropertyPath::Iri(NamedNode::new_unchecked("http://example.org/p1"));
        let p2 = PropertyPath::Iri(NamedNode::new_unchecked("http://example.org/p2"));
        let seq = PropertyPath::Sequence(Box::new(p1), Box::new(p2));

        let display = seq.to_string();
        assert!(display.contains('/'), "Sequence path display contains '/'");
    }

    #[test]
    fn test_property_path_alternative_display() {
        use oxirs_arq::algebra::PropertyPath;

        let p1 = PropertyPath::Iri(NamedNode::new_unchecked("http://example.org/p1"));
        let p2 = PropertyPath::Iri(NamedNode::new_unchecked("http://example.org/p2"));
        let alt = PropertyPath::Alternative(Box::new(p1), Box::new(p2));

        let display = alt.to_string();
        assert!(
            display.contains('|'),
            "Alternative path display contains '|'"
        );
    }

    #[test]
    fn test_property_path_zero_or_more_display() {
        use oxirs_arq::algebra::PropertyPath;

        let p = PropertyPath::Iri(NamedNode::new_unchecked("http://example.org/knows"));
        let zero_or_more = PropertyPath::ZeroOrMore(Box::new(p));

        let display = zero_or_more.to_string();
        assert!(
            display.contains('*'),
            "ZeroOrMore path display contains '*'"
        );
    }
}

// ============================================================
// aggregate_extended_tests — 7 tests for SUM, MIN, MAX, AVG,
// SAMPLE, GROUP_CONCAT, and DISTINCT COUNT aggregates
// ============================================================
#[cfg(test)]
mod aggregate_extended_tests {
    use super::*;
    use oxirs_arq::Aggregate;

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
