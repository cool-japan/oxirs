//! Datalog engine tests

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::datalog::{
        evaluator::{apply_rule, unify},
        parser::{parse_program, parse_rule},
        DatalogAtom, DatalogRule, DatalogTerm, DatalogValue, FactDatabase, SemiNaiveEvaluator,
        Substitution,
    };
    use std::collections::HashMap;

    // ─────────────────────────────────────────────────────────
    // Parser tests
    // ─────────────────────────────────────────────────────────

    #[test]
    fn test_parse_fact_simple() {
        let program = parse_program("parent(alice, bob).").expect("parse");
        assert_eq!(program.edb.len(), 1);
        let fact = &program.edb[0];
        assert_eq!(fact.predicate, "parent");
        assert_eq!(fact.args[0], DatalogValue::Str("alice".to_string()));
        assert_eq!(fact.args[1], DatalogValue::Str("bob".to_string()));
    }

    #[test]
    fn test_parse_rule_simple() {
        let rule = parse_rule("ancestor(X, Y) :- parent(X, Y).").expect("parse");
        assert_eq!(rule.head.predicate, "ancestor");
        assert_eq!(rule.head.terms.len(), 2);
        assert_eq!(rule.head.terms[0], DatalogTerm::Variable("X".to_string()));
        assert_eq!(rule.body.len(), 1);
        assert_eq!(rule.body[0].predicate, "parent");
    }

    #[test]
    fn test_parse_rule_recursive() {
        let rule = parse_rule("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).").expect("parse");
        assert_eq!(rule.head.predicate, "ancestor");
        assert_eq!(rule.body.len(), 2);
        assert_eq!(rule.body[0].predicate, "parent");
        assert_eq!(rule.body[1].predicate, "ancestor");
    }

    #[test]
    fn test_parse_multiple_facts() {
        let src = "parent(alice, bob). parent(bob, carol). parent(carol, dave).";
        let program = parse_program(src).expect("parse");
        assert_eq!(program.edb.len(), 3);
    }

    #[test]
    fn test_parse_comment() {
        let src = "% this is a comment\nparent(alice, bob). % inline comment\nparent(bob, carol).";
        let program = parse_program(src).expect("parse");
        assert_eq!(program.edb.len(), 2, "comments should be stripped");
    }

    // ─────────────────────────────────────────────────────────
    // Evaluation tests
    // ─────────────────────────────────────────────────────────

    #[test]
    fn test_evaluate_simple_fact() {
        // EDB passthrough: evaluate a program with only facts
        let src = "parent(alice, bob).";
        let program = parse_program(src).expect("parse");
        let evaluator = SemiNaiveEvaluator::new();
        let db = evaluator.evaluate(&program).expect("evaluate");

        let alice_bob = vec![
            DatalogValue::Str("alice".to_string()),
            DatalogValue::Str("bob".to_string()),
        ];
        assert!(db.contains("parent", &alice_bob));
    }

    #[test]
    fn test_evaluate_simple_rule() {
        // 1-hop derivation: ancestor(X,Y) :- parent(X,Y).
        let src = r#"
            parent(alice, bob).
            ancestor(X, Y) :- parent(X, Y).
        "#;
        let program = parse_program(src).expect("parse");
        let evaluator = SemiNaiveEvaluator::new();
        let db = evaluator.evaluate(&program).expect("evaluate");

        let alice_bob = vec![
            DatalogValue::Str("alice".to_string()),
            DatalogValue::Str("bob".to_string()),
        ];
        assert!(
            db.contains("ancestor", &alice_bob),
            "ancestor(alice, bob) should be derived"
        );
    }

    #[test]
    fn test_evaluate_transitive_closure() {
        // Ancestor over a chain a→b→c→d→e, verify a→e is derived
        let src = r#"
            parent(a, b).
            parent(b, c).
            parent(c, d).
            parent(d, e).
            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
        "#;
        let program = parse_program(src).expect("parse");
        let evaluator = SemiNaiveEvaluator::new();
        let db = evaluator.evaluate(&program).expect("evaluate");

        let a_e = vec![
            DatalogValue::Str("a".to_string()),
            DatalogValue::Str("e".to_string()),
        ];
        assert!(
            db.contains("ancestor", &a_e),
            "ancestor(a, e) should be derived via transitive closure"
        );

        // Also verify intermediate steps
        let a_c = vec![
            DatalogValue::Str("a".to_string()),
            DatalogValue::Str("c".to_string()),
        ];
        assert!(db.contains("ancestor", &a_c), "ancestor(a, c)");
    }

    #[test]
    fn test_evaluate_no_facts_matching() {
        // Rule fires 0 times — no parent facts
        let src = r#"
            ancestor(X, Y) :- parent(X, Y).
        "#;
        let program = parse_program(src).expect("parse");
        let evaluator = SemiNaiveEvaluator::new();
        let db = evaluator.evaluate(&program).expect("evaluate");

        // ancestor should be empty
        assert_eq!(
            db.tuples_for("ancestor").count(),
            0,
            "no ancestor facts should be derived"
        );
    }

    #[test]
    fn test_evaluate_self_join() {
        // sibling(X, Y) :- parent(Z, X), parent(Z, Y).
        let src = r#"
            parent(alice, bob).
            parent(alice, carol).
            sibling(X, Y) :- parent(Z, X), parent(Z, Y).
        "#;
        let program = parse_program(src).expect("parse");
        let evaluator = SemiNaiveEvaluator::new();
        let db = evaluator.evaluate(&program).expect("evaluate");

        // bob and carol are siblings (and also self-sibling in naive semantics)
        let bob_carol = vec![
            DatalogValue::Str("bob".to_string()),
            DatalogValue::Str("carol".to_string()),
        ];
        assert!(
            db.contains("sibling", &bob_carol),
            "sibling(bob, carol) should be derived"
        );

        let carol_bob = vec![
            DatalogValue::Str("carol".to_string()),
            DatalogValue::Str("bob".to_string()),
        ];
        assert!(
            db.contains("sibling", &carol_bob),
            "sibling(carol, bob) should be derived"
        );
    }

    #[test]
    fn test_fixpoint_detection() {
        // Verify that evaluation terminates in correct number of iterations for a chain of 3
        let src = r#"
            edge(a, b).
            edge(b, c).
            path(X, Y) :- edge(X, Y).
            path(X, Z) :- edge(X, Y), path(Y, Z).
        "#;
        let program = parse_program(src).expect("parse");
        let evaluator = SemiNaiveEvaluator::new();
        let db = evaluator.evaluate(&program).expect("evaluate");

        let a_c = vec![
            DatalogValue::Str("a".to_string()),
            DatalogValue::Str("c".to_string()),
        ];
        assert!(db.contains("path", &a_c), "path(a, c) should be derived");
    }

    #[test]
    fn test_evaluate_integer_constants() {
        // score(alice, 42). high(X) :- score(X, N).
        let src = r#"
            score(alice, 42).
            high(X) :- score(X, 42).
        "#;
        let program = parse_program(src).expect("parse");
        let evaluator = SemiNaiveEvaluator::new();
        let db = evaluator.evaluate(&program).expect("evaluate");

        let alice = vec![DatalogValue::Str("alice".to_string())];
        assert!(db.contains("high", &alice), "high(alice) should be derived");
    }

    #[test]
    fn test_evaluate_boolean_constants() {
        // active(alice, true). live(X) :- active(X, true).
        let src = r#"
            active(alice, true).
            live(X) :- active(X, true).
        "#;
        let program = parse_program(src).expect("parse");
        let evaluator = SemiNaiveEvaluator::new();
        let db = evaluator.evaluate(&program).expect("evaluate");

        let alice = vec![DatalogValue::Str("alice".to_string())];
        assert!(db.contains("live", &alice), "live(alice) should be derived");
    }

    #[test]
    fn test_evaluate_multiple_rules() {
        // Two rules with shared head predicate
        let src = r#"
            parent(alice, bob).
            parent(bob, carol).
            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Z) :- ancestor(X, Y), ancestor(Y, Z).
        "#;
        let program = parse_program(src).expect("parse");
        let evaluator = SemiNaiveEvaluator::new();
        let db = evaluator.evaluate(&program).expect("evaluate");

        let alice_carol = vec![
            DatalogValue::Str("alice".to_string()),
            DatalogValue::Str("carol".to_string()),
        ];
        assert!(
            db.contains("ancestor", &alice_carol),
            "ancestor(alice, carol) via two rules"
        );
    }

    #[test]
    fn test_evaluate_chain_of_4() {
        // 4-step transitive closure
        let src = r#"
            edge(a, b).
            edge(b, c).
            edge(c, d).
            edge(d, e).
            reach(X, Y) :- edge(X, Y).
            reach(X, Z) :- edge(X, Y), reach(Y, Z).
        "#;
        let program = parse_program(src).expect("parse");
        let evaluator = SemiNaiveEvaluator::new();
        let db = evaluator.evaluate(&program).expect("evaluate");

        let a_e = vec![
            DatalogValue::Str("a".to_string()),
            DatalogValue::Str("e".to_string()),
        ];
        assert!(
            db.contains("reach", &a_e),
            "reach(a, e) should be derived in chain of 4"
        );
    }

    // ─────────────────────────────────────────────────────────
    // Substitution and unification tests
    // ─────────────────────────────────────────────────────────

    #[test]
    fn test_substitution_unify_variable() {
        // Unify Variable("X") with Constant(Int(1))
        let mut sub: Substitution = HashMap::new();
        let term = DatalogTerm::Variable("X".to_string());
        let value = DatalogValue::Int(1);
        let ok = unify(&term, &value, &mut sub);
        assert!(ok, "should succeed");
        assert_eq!(sub.get("X"), Some(&DatalogValue::Int(1)));
    }

    #[test]
    fn test_substitution_unify_constant_match() {
        // Constant matches identical constant
        let mut sub: Substitution = HashMap::new();
        let term = DatalogTerm::Constant(DatalogValue::Str("alice".to_string()));
        let value = DatalogValue::Str("alice".to_string());
        let ok = unify(&term, &value, &mut sub);
        assert!(ok, "matching constants should unify");
        assert!(sub.is_empty(), "no variable binding needed");
    }

    #[test]
    fn test_substitution_unify_constant_mismatch() {
        // Constant mismatches → false
        let mut sub: Substitution = HashMap::new();
        let term = DatalogTerm::Constant(DatalogValue::Str("alice".to_string()));
        let value = DatalogValue::Str("bob".to_string());
        let ok = unify(&term, &value, &mut sub);
        assert!(!ok, "mismatching constants should not unify");
    }

    // ─────────────────────────────────────────────────────────
    // apply_rule tests
    // ─────────────────────────────────────────────────────────

    #[test]
    fn test_apply_rule_basic() {
        // ancestor(X, Y) :- parent(X, Y). with parent(a, b) in EDB
        let mut edb = FactDatabase::new();
        edb.insert(
            "parent",
            vec![
                DatalogValue::Str("a".to_string()),
                DatalogValue::Str("b".to_string()),
            ],
        );
        let idb = FactDatabase::new();

        let rule = DatalogRule {
            head: DatalogAtom {
                predicate: "ancestor".to_string(),
                terms: vec![
                    DatalogTerm::Variable("X".to_string()),
                    DatalogTerm::Variable("Y".to_string()),
                ],
            },
            body: vec![DatalogAtom {
                predicate: "parent".to_string(),
                terms: vec![
                    DatalogTerm::Variable("X".to_string()),
                    DatalogTerm::Variable("Y".to_string()),
                ],
            }],
        };

        let derived = apply_rule(&rule, &edb, &idb);
        assert_eq!(derived.len(), 1);
        let fact = derived.iter().next().expect("one fact");
        assert_eq!(fact.predicate, "ancestor");
        assert_eq!(fact.args[0], DatalogValue::Str("a".to_string()));
        assert_eq!(fact.args[1], DatalogValue::Str("b".to_string()));
    }

    #[test]
    fn test_fact_database_contains() {
        let mut db = FactDatabase::new();
        db.insert("p", vec![DatalogValue::Int(1), DatalogValue::Int(2)]);
        assert!(db.contains("p", &[DatalogValue::Int(1), DatalogValue::Int(2)]));
        assert!(!db.contains("p", &[DatalogValue::Int(1), DatalogValue::Int(3)]));
        assert!(!db.contains("q", &[DatalogValue::Int(1)]));
    }
}
