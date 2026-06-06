//! SPARQL Compliance Tests — UNION semantics (`union_tests`).
//!
//! Split out of `sparql_compliance.rs` to keep each integration test source
//! file below the 2000-line refactoring policy. Shared fixtures live in
//! `tests/common/mod.rs`.

mod common;

use common::create_test_dataset;
use oxirs_arq::algebra::Term;
use oxirs_arq::{Algebra, Literal, QueryExecutor, TriplePattern, Variable};
use oxirs_core::model::NamedNode;

// ============================================================
// union_tests — 4 tests for UNION semantics
// ============================================================
#[cfg(test)]
mod union_tests {
    use super::*;

    #[test]
    fn test_union_both_sides_match() {
        let dataset = create_test_dataset();
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
        let dataset = create_test_dataset();
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
        let dataset = create_test_dataset();
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
        let dataset = create_test_dataset();
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
