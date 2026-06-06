//! SPARQL Compliance Tests — LIMIT/OFFSET (`limit_offset_tests`) and ORDER BY
//! (`order_by_tests`).
//!
//! Split out of `sparql_compliance.rs` to keep each integration test source
//! file below the 2000-line refactoring policy. Shared fixtures live in
//! `tests/common/mod.rs`.

mod common;

use common::create_test_dataset;
use oxirs_arq::algebra::Term;
use oxirs_arq::{Algebra, Expression, QueryExecutor, TriplePattern, Variable};
use oxirs_core::model::NamedNode;

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
        let dataset = create_test_dataset();
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
        let dataset = create_test_dataset();
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
        let dataset = create_test_dataset();
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
        let dataset = create_test_dataset();
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
        let dataset = create_test_dataset();
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
        let dataset = create_test_dataset();
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
        let dataset = create_test_dataset();
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
        let dataset = create_test_dataset();
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
        let dataset = create_test_dataset();
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
        let dataset = create_test_dataset();
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
