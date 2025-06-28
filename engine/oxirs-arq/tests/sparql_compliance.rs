//! SPARQL Compliance Tests
//!
//! This module provides basic SPARQL compliance testing while we work
//! towards full W3C test suite integration.

use oxirs_arq::algebra::Term;
use oxirs_arq::{
    Algebra, BinaryOperator, Expression, Iri, Literal, QueryExecutor, Solution, TriplePattern,
    Variable,
};
use std::collections::HashMap;

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
            Term::Iri(Iri("http://example.org/alice".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            Term::Literal(Literal {
                value: "Alice".to_string(),
                language: None,
                datatype: None,
            }),
        );

        dataset.add_triple(
            Term::Iri(Iri("http://example.org/alice".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/age".to_string())),
            Term::Literal(Literal {
                value: "30".to_string(),
                language: None,
                datatype: Some(Iri("http://www.w3.org/2001/XMLSchema#integer".to_string())),
            }),
        );

        dataset.add_triple(
            Term::Iri(Iri("http://example.org/bob".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            Term::Literal(Literal {
                value: "Bob".to_string(),
                language: None,
                datatype: None,
            }),
        );

        dataset.add_triple(
            Term::Iri(Iri("http://example.org/bob".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/age".to_string())),
            Term::Literal(Literal {
                value: "25".to_string(),
                language: None,
                datatype: Some(Iri("http://www.w3.org/2001/XMLSchema#integer".to_string())),
            }),
        );

        dataset
    }

    #[test]
    fn test_basic_bgp() {
        let dataset = create_test_dataset();
        let executor = QueryExecutor::new();

        // Query: ?person foaf:name ?name
        let algebra = Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable("person".to_string()),
            predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            object: Term::Variable("name".to_string()),
        }]);

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        assert_eq!(solution.len(), 2);

        // Check that we got both Alice and Bob
        let names: Vec<String> = solution
            .iter()
            .filter_map(|binding| {
                binding.get("name").and_then(|term| match term {
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
        let executor = QueryExecutor::new();

        // Query: ?person foaf:age ?age . FILTER(?age > 25)
        let algebra = Algebra::Filter {
            pattern: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable("person".to_string()),
                predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/age".to_string())),
                object: Term::Variable("age".to_string()),
            }])),
            condition: Expression::Binary {
                op: BinaryOperator::Greater,
                left: Box::new(Expression::Variable("age".to_string())),
                right: Box::new(Expression::Literal(Literal {
                    value: "25".to_string(),
                    language: None,
                    datatype: Some(Iri("http://www.w3.org/2001/XMLSchema#integer".to_string())),
                })),
            },
        };

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        // Should only return Alice (age 30)
        assert_eq!(solution.len(), 1);

        let person = solution[0].get("person").unwrap();
        match person {
            Term::Iri(iri) => assert_eq!(iri.0, "http://example.org/alice"),
            _ => panic!("Expected IRI"),
        }
    }

    #[test]
    fn test_optional_pattern() {
        let dataset = create_test_dataset();
        let executor = QueryExecutor::new();

        // Add a person without age
        let mut dataset = dataset;
        dataset.add_triple(
            Term::Iri(Iri("http://example.org/charlie".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            Term::Literal(Literal {
                value: "Charlie".to_string(),
                language: None,
                datatype: None,
            }),
        );

        // Query: ?person foaf:name ?name . OPTIONAL { ?person foaf:age ?age }
        let algebra = Algebra::LeftJoin {
            left: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable("person".to_string()),
                predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
                object: Term::Variable("name".to_string()),
            }])),
            right: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable("person".to_string()),
                predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/age".to_string())),
                object: Term::Variable("age".to_string()),
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
                    .get("name")
                    .and_then(|term| match term {
                        Term::Literal(lit) => Some(lit.value.as_str()),
                        _ => None,
                    })
                    .map_or(false, |name| name == "Charlie")
            })
            .unwrap();

        assert!(charlie_binding.contains_key("name"));
        assert!(!charlie_binding.contains_key("age"));
    }

    #[test]
    fn test_union_pattern() {
        let dataset = create_test_dataset();
        let executor = QueryExecutor::new();

        // Query: { ?s foaf:name "Alice" } UNION { ?s foaf:age "25"^^xsd:integer }
        let algebra = Algebra::Union {
            left: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable("s".to_string()),
                predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
                object: Term::Literal(Literal {
                    value: "Alice".to_string(),
                    language: None,
                    datatype: None,
                }),
            }])),
            right: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable("s".to_string()),
                predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/age".to_string())),
                object: Term::Literal(Literal {
                    value: "25".to_string(),
                    language: None,
                    datatype: Some(Iri("http://www.w3.org/2001/XMLSchema#integer".to_string())),
                }),
            }])),
        };

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        // Should return Alice and Bob (Bob has age 25)
        assert_eq!(solution.len(), 2);

        let subjects: Vec<String> = solution
            .iter()
            .filter_map(|binding| {
                binding.get("s").and_then(|term| match term {
                    Term::Iri(iri) => Some(iri.0.clone()),
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
        let executor = QueryExecutor::new();

        // Query: SELECT ?name WHERE { ?person foaf:name ?name ; foaf:age ?age }
        let algebra = Algebra::Project {
            pattern: Box::new(Algebra::Bgp(vec![
                TriplePattern {
                    subject: Term::Variable("person".to_string()),
                    predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
                    object: Term::Variable("name".to_string()),
                },
                TriplePattern {
                    subject: Term::Variable("person".to_string()),
                    predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/age".to_string())),
                    object: Term::Variable("age".to_string()),
                },
            ])),
            variables: vec!["name".to_string()],
        };

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        // Check that only 'name' variable is in results
        for binding in &solution {
            assert!(binding.contains_key("name"));
            assert!(!binding.contains_key("person"));
            assert!(!binding.contains_key("age"));
        }
    }

    #[test]
    fn test_distinct() {
        let dataset = create_test_dataset();
        let executor = QueryExecutor::new();

        // Add duplicate data
        let mut dataset = dataset;
        dataset.add_triple(
            Term::Iri(Iri("http://example.org/alice2".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
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
                    subject: Term::Variable("person".to_string()),
                    predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
                    object: Term::Variable("name".to_string()),
                }])),
                variables: vec!["name".to_string()],
            }),
        };

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        // Should have only 2 distinct names (Alice and Bob)
        assert_eq!(solution.len(), 2);
    }

    #[test]
    fn test_order_by() {
        let dataset = create_test_dataset();
        let executor = QueryExecutor::new();

        // Query: SELECT ?person ?age WHERE { ?person foaf:age ?age } ORDER BY ?age
        let algebra = Algebra::OrderBy {
            pattern: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable("person".to_string()),
                predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/age".to_string())),
                object: Term::Variable("age".to_string()),
            }])),
            conditions: vec![oxirs_arq::OrderCondition {
                expr: Expression::Variable("age".to_string()),
                ascending: true,
            }],
        };

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        // Bob (25) should come before Alice (30)
        assert_eq!(solution.len(), 2);

        let first_person = solution[0].get("person").unwrap();
        match first_person {
            Term::Iri(iri) => assert_eq!(iri.0, "http://example.org/bob"),
            _ => panic!("Expected IRI"),
        }

        let second_person = solution[1].get("person").unwrap();
        match second_person {
            Term::Iri(iri) => assert_eq!(iri.0, "http://example.org/alice"),
            _ => panic!("Expected IRI"),
        }
    }

    #[test]
    fn test_limit_offset() {
        let dataset = create_test_dataset();
        let executor = QueryExecutor::new();

        // Query: SELECT ?person WHERE { ?person foaf:name ?name } LIMIT 1 OFFSET 1
        let algebra = Algebra::Slice {
            pattern: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable("person".to_string()),
                predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
                object: Term::Variable("name".to_string()),
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
        let executor = QueryExecutor::new();

        // Query: SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }
        let algebra = Algebra::Group {
            pattern: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable("s".to_string()),
                predicate: Term::Variable("p".to_string()),
                object: Term::Variable("o".to_string()),
            }])),
            variables: vec![], // No grouping variables = single group
            aggregates: vec![(
                "count".to_string(),
                Aggregate::Count {
                    distinct: false,
                    expr: None, // COUNT(*)
                },
            )],
        };

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();

        println!("Solution length: {}", solution.len());
        for (i, binding) in solution.iter().enumerate() {
            println!("Binding {}: {:?}", i, binding);
        }

        assert_eq!(solution.len(), 1);
        let count = solution[0].get("count").unwrap();
        match count {
            Term::Literal(lit) => assert_eq!(lit.value, "4"), // 4 triples in test dataset
            _ => panic!("Expected literal"),
        }
    }

    #[test]
    fn test_group_by_with_count() {
        let dataset = super::basic_tests::create_test_dataset();
        let executor = QueryExecutor::new();

        // Add more data for grouping
        let mut dataset = dataset;
        dataset.add_triple(
            Term::Iri(Iri("http://example.org/alice".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/knows".to_string())),
            Term::Iri(Iri("http://example.org/bob".to_string())),
        );

        // Query: SELECT ?person (COUNT(*) as ?count) WHERE { ?person ?p ?o } GROUP BY ?person
        let algebra = Algebra::Group {
            pattern: Box::new(Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable("person".to_string()),
                predicate: Term::Variable("p".to_string()),
                object: Term::Variable("o".to_string()),
            }])),
            variables: vec![GroupCondition {
                expr: Expression::Variable("person".to_string()),
                alias: None,
            }],
            aggregates: vec![(
                "count".to_string(),
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
                    .get("person")
                    .and_then(|term| match term {
                        Term::Iri(iri) => Some(iri.0.as_str()),
                        _ => None,
                    })
                    .map_or(false, |iri| iri.contains("alice"))
            })
            .unwrap();

        let alice_count = alice_binding.get("count").unwrap();
        match alice_count {
            Term::Literal(lit) => assert_eq!(lit.value, "3"),
            _ => panic!("Expected literal"),
        }
    }
}
