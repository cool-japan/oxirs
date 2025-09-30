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
