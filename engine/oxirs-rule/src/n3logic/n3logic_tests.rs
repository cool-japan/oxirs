//! # N3 Logic Tests
//!
//! All unit tests for the N3 parser, engine, built-ins, and type helpers.

#[cfg(test)]
mod tests {
    use super::super::n3logic_evaluator::N3Engine;
    use super::super::n3logic_parser::N3Parser;
    use super::super::n3logic_types::{Bindings, N3BuiltIn, N3Formula, N3Rule, N3Term, Triple};

    // Helper: make a simple triple-pattern formula
    fn tf(s: &str, p: &str, o: &str) -> N3Formula {
        N3Formula::Triple {
            subject: if let Some(stripped) = s.strip_prefix('?') {
                N3Term::Variable(stripped.to_string())
            } else {
                N3Term::Iri(s.to_string())
            },
            predicate: if let Some(stripped) = p.strip_prefix('?') {
                N3Term::Variable(stripped.to_string())
            } else {
                N3Term::Iri(p.to_string())
            },
            object: if let Some(stripped) = o.strip_prefix('?') {
                N3Term::Variable(stripped.to_string())
            } else {
                N3Term::Iri(o.to_string())
            },
        }
    }

    // ── Parser tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_parse_simple_rule() -> Result<(), Box<dyn std::error::Error>> {
        let rules = N3Parser::parse("{ ?a :hasAge ?n } => { ?a a :Adult } .")?;
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].antecedent.len(), 1);
        assert_eq!(rules[0].consequent.len(), 1);
        Ok(())
    }

    #[test]
    fn test_parse_rule_method() -> Result<(), Box<dyn std::error::Error>> {
        let rule = N3Parser::parse_rule("{ ?a :hasAge ?n } => { ?a a :Adult }")?;
        assert_eq!(rule.antecedent.len(), 1);
        assert_eq!(rule.consequent.len(), 1);
        Ok(())
    }

    #[test]
    fn test_parse_rule_antecedent_has_variable_subject() -> Result<(), Box<dyn std::error::Error>> {
        let rules = N3Parser::parse("{ ?x :knows ?y } => { ?x :acquaintance ?y } .")?;
        match &rules[0].antecedent[0] {
            N3Formula::Triple { subject, .. } => {
                assert!(matches!(subject, N3Term::Variable(v) if v == "x"))
            }
            _ => panic!("expected triple"),
        }
        Ok(())
    }

    #[test]
    fn test_parse_rule_antecedent_has_variable_object() -> Result<(), Box<dyn std::error::Error>> {
        let rules = N3Parser::parse("{ ?x :knows ?y } => { ?x :acquaintance ?y } .")?;
        match &rules[0].antecedent[0] {
            N3Formula::Triple { object, .. } => {
                assert!(matches!(object, N3Term::Variable(v) if v == "y"))
            }
            _ => panic!("expected triple"),
        }
        Ok(())
    }

    #[test]
    fn test_parse_multiple_antecedents() -> Result<(), Box<dyn std::error::Error>> {
        let n3 = "{ ?a :hasAge ?n . ?a a :Person } => { ?a :isAdult true } .";
        let rules = N3Parser::parse(n3)?;
        assert_eq!(rules[0].antecedent.len(), 2);
        Ok(())
    }

    #[test]
    fn test_parse_multiple_rules() -> Result<(), Box<dyn std::error::Error>> {
        let n3 =
            "{ ?a :hasAge ?n } => { ?a a :Person } . { ?a a :Person } => { ?a :hasType :Human } .";
        let rules = N3Parser::parse(n3)?;
        assert_eq!(rules.len(), 2);
        Ok(())
    }

    #[test]
    fn test_parse_forall_declaration() -> Result<(), Box<dyn std::error::Error>> {
        let n3 = "@forAll :x, :n . { :x :hasAge :n } => { :x a :Adult } .";
        let rules = N3Parser::parse(n3)?;
        assert!(!rules[0].universals.is_empty());
        Ok(())
    }

    #[test]
    fn test_parse_forsome_declaration() -> Result<(), Box<dyn std::error::Error>> {
        let n3 = "@forSome :y . { ?x :knows :y } => { ?x :hasFriend :y } .";
        let rules = N3Parser::parse(n3)?;
        assert!(!rules[0].existentials.is_empty());
        Ok(())
    }

    #[test]
    fn test_parse_universal_count() -> Result<(), Box<dyn std::error::Error>> {
        let n3 = "@forAll :x, :y . { :x :p :y } => { :y :q :x } .";
        let rules = N3Parser::parse(n3)?;
        assert_eq!(rules[0].universals.len(), 2);
        Ok(())
    }

    #[test]
    fn test_parse_iri_terms() -> Result<(), Box<dyn std::error::Error>> {
        let n3 = "{ <http://a.org/x> <http://a.org/p> <http://a.org/y> } => { <http://a.org/x> <http://a.org/q> <http://a.org/y> } .";
        let rules = N3Parser::parse(n3)?;
        match &rules[0].antecedent[0] {
            N3Formula::Triple { subject, .. } => {
                assert!(matches!(subject, N3Term::Iri(s) if s == "http://a.org/x"))
            }
            _ => panic!("expected triple"),
        }
        Ok(())
    }

    #[test]
    fn test_parse_string_literal() -> Result<(), Box<dyn std::error::Error>> {
        let rules = N3Parser::parse(r#"{ ?x :name "Alice" } => { ?x a :Named } ."#)?;
        match &rules[0].antecedent[0] {
            N3Formula::Triple { object, .. } => {
                assert!(matches!(object, N3Term::Literal { value, .. } if value == "Alice"))
            }
            _ => panic!("expected triple"),
        }
        Ok(())
    }

    #[test]
    fn test_parse_number_literal() -> Result<(), Box<dyn std::error::Error>> {
        let rules = N3Parser::parse("{ ?x :age 42 } => { ?x a :MiddleAged } .")?;
        match &rules[0].antecedent[0] {
            N3Formula::Triple { object, .. } => {
                assert!(matches!(object, N3Term::Literal { value, .. } if value == "42"))
            }
            _ => panic!("expected triple"),
        }
        Ok(())
    }

    #[test]
    fn test_parse_blank_node() -> Result<(), Box<dyn std::error::Error>> {
        let rules = N3Parser::parse("{ _:b1 :p _:b2 } => { _:b1 :q _:b2 } .")?;
        match &rules[0].antecedent[0] {
            N3Formula::Triple { subject, .. } => {
                assert!(matches!(subject, N3Term::BlankNode(s) if s == "b1"))
            }
            _ => panic!("expected triple"),
        }
        Ok(())
    }

    #[test]
    fn test_parse_rdf_type_shorthand() -> Result<(), Box<dyn std::error::Error>> {
        let rules = N3Parser::parse("{ ?x a :Person } => { ?x a :Being } .")?;
        match &rules[0].antecedent[0] {
            N3Formula::Triple { predicate, .. } => {
                assert!(matches!(predicate, N3Term::Iri(p) if p.contains("type")))
            }
            _ => panic!("expected triple"),
        }
        Ok(())
    }

    #[test]
    fn test_parse_math_greater_than() -> Result<(), Box<dyn std::error::Error>> {
        let rules = N3Parser::parse("{ ?x math:greaterThan 18 } => { ?x a :Adult } .")?;
        assert!(matches!(
            &rules[0].antecedent[0],
            N3Formula::BuiltIn(N3BuiltIn::MathGreaterThan { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_parse_math_less_than() -> Result<(), Box<dyn std::error::Error>> {
        let rules = N3Parser::parse("{ ?x math:lessThan 18 } => { ?x a :Minor } .")?;
        assert!(matches!(
            &rules[0].antecedent[0],
            N3Formula::BuiltIn(N3BuiltIn::MathLessThan { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_parse_log_equal() -> Result<(), Box<dyn std::error::Error>> {
        let rules = N3Parser::parse("{ ?x log:equal ?y } => { ?x :same ?y } .")?;
        assert!(matches!(
            &rules[0].antecedent[0],
            N3Formula::BuiltIn(N3BuiltIn::LogEqual { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_parse_log_not_equal() -> Result<(), Box<dyn std::error::Error>> {
        let rules = N3Parser::parse("{ ?x log:notEqual ?y } => { ?x :different ?y } .")?;
        assert!(matches!(
            &rules[0].antecedent[0],
            N3Formula::BuiltIn(N3BuiltIn::LogNotEqual { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_parse_string_length_builtin() -> Result<(), Box<dyn std::error::Error>> {
        let rules = N3Parser::parse(r#"{ ?x string:length ?n } => { ?x :hasLength ?n } ."#)?;
        assert!(matches!(
            &rules[0].antecedent[0],
            N3Formula::BuiltIn(N3BuiltIn::StringLength { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_parse_empty_document() -> Result<(), Box<dyn std::error::Error>> {
        let rules = N3Parser::parse("")?;
        assert_eq!(rules.len(), 0);
        Ok(())
    }

    #[test]
    fn test_parse_with_comments() -> Result<(), Box<dyn std::error::Error>> {
        let rules = N3Parser::parse("# comment\n{ ?x :p ?y } => { ?x :q ?y } .")?;
        assert_eq!(rules.len(), 1);
        Ok(())
    }

    #[test]
    fn test_parse_consequent_variable_order() -> Result<(), Box<dyn std::error::Error>> {
        let rules = N3Parser::parse("{ ?a :parent ?b } => { ?b :child ?a } .")?;
        match &rules[0].consequent[0] {
            N3Formula::Triple { subject, .. } => {
                assert!(matches!(subject, N3Term::Variable(v) if v == "b"))
            }
            _ => panic!("expected triple"),
        }
        Ok(())
    }

    #[test]
    fn test_parse_with_prefix_declaration() -> Result<(), Box<dyn std::error::Error>> {
        let n3 = "@prefix ex: <http://example.org/> . { ?x ex:p ?y } => { ?x ex:q ?y } .";
        let rules = N3Parser::parse(n3)?;
        assert_eq!(rules.len(), 1);
        Ok(())
    }

    // ── Engine: basic forward chaining ───────────────────────────────────────

    #[test]
    fn test_engine_simple_derivation() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":p", "?y")],
            vec![tf("?x", ":q", "?y")],
        ));
        engine.assert_fact(Triple::new("a", ":p", "b"));
        let facts = engine.run()?;
        assert!(facts
            .iter()
            .any(|t| t.subject == "a" && t.predicate == ":q" && t.object == "b"));
        Ok(())
    }

    #[test]
    fn test_engine_derive_inverse_relation() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":parent", "?y")],
            vec![tf("?y", ":child", "?x")],
        ));
        engine.assert_fact(Triple::new("alice", ":parent", "bob"));
        let facts = engine.run()?;
        assert!(facts
            .iter()
            .any(|t| t.subject == "bob" && t.predicate == ":child" && t.object == "alice"));
        Ok(())
    }

    #[test]
    fn test_engine_chain_two_rules() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":parent", "?y")],
            vec![tf("?x", ":ancestor", "?y")],
        ));
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":ancestor", "?y"), tf("?y", ":parent", "?z")],
            vec![tf("?x", ":ancestor", "?z")],
        ));
        engine.assert_fact(Triple::new("alice", ":parent", "bob"));
        engine.assert_fact(Triple::new("bob", ":parent", "carol"));
        let facts = engine.run()?;
        assert!(
            facts
                .iter()
                .any(|t| t.subject == "alice" && t.predicate == ":ancestor" && t.object == "carol"),
            "got: {:?}",
            facts
        );
        Ok(())
    }

    #[test]
    fn test_engine_bounded_terminates() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":linked", "?y")],
            vec![tf("?y", ":linked", "?x")],
        ));
        engine.assert_fact(Triple::new("a", ":linked", "b"));
        let facts = engine.run_bounded(5)?;
        assert!(!facts.is_empty());
        Ok(())
    }

    #[test]
    fn test_engine_fixpoint_stable() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":p", "?y")],
            vec![tf("?x", ":q", "?y")],
        ));
        engine.assert_fact(Triple::new("x1", ":p", "y1"));
        let c1 = engine.run_bounded(1)?.len();
        let mut engine2 = N3Engine::new();
        engine2.add_rule(N3Rule::new(
            vec![tf("?x", ":p", "?y")],
            vec![tf("?x", ":q", "?y")],
        ));
        engine2.assert_fact(Triple::new("x1", ":p", "y1"));
        let c100 = engine2.run_bounded(100)?.len();
        assert_eq!(c1, c100);
        Ok(())
    }

    #[test]
    fn test_engine_is_derivable_from_fact() {
        let mut engine = N3Engine::new();
        engine.assert_fact(Triple::new("a", ":p", "b"));
        assert!(engine.is_derivable(&Triple::new("a", ":p", "b")));
    }

    #[test]
    fn test_engine_is_not_derivable() {
        let engine = N3Engine::new();
        assert!(!engine.is_derivable(&Triple::new("a", ":p", "b")));
    }

    #[test]
    fn test_engine_is_derivable_via_rule() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?a", ":knows", "?b")],
            vec![tf("?a", ":acquaintance", "?b")],
        ));
        engine.assert_fact(Triple::new("alice", ":knows", "bob"));
        assert!(engine.is_derivable(&Triple::new("alice", ":acquaintance", "bob")));
        Ok(())
    }

    #[test]
    fn test_engine_is_not_derivable_no_match() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?a", ":knows", "?b")],
            vec![tf("?a", ":acquaintance", "?b")],
        ));
        engine.assert_fact(Triple::new("alice", ":knows", "bob"));
        assert!(!engine.is_derivable(&Triple::new("carol", ":acquaintance", "dave")));
        Ok(())
    }

    #[test]
    fn test_engine_no_rules_returns_facts() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = N3Engine::new();
        engine.assert_fact(Triple::new("a", ":p", "b"));
        let facts = engine.run()?;
        assert_eq!(facts.len(), 1);
        Ok(())
    }

    #[test]
    fn test_engine_no_matching_rule() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":never", "?y")],
            vec![tf("?x", ":derived", "?y")],
        ));
        engine.assert_fact(Triple::new("a", ":different", "b"));
        let facts = engine.run()?;
        assert_eq!(facts.len(), 1);
        Ok(())
    }

    #[test]
    fn test_engine_multiple_facts_derived() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":parent", "?y")],
            vec![tf("?x", ":ancestor", "?y")],
        ));
        engine.assert_fact(Triple::new("alice", ":parent", "bob"));
        engine.assert_fact(Triple::new("carol", ":parent", "dave"));
        let facts = engine.run()?;
        assert!(facts
            .iter()
            .any(|t| t.subject == "alice" && t.predicate == ":ancestor"));
        assert!(facts
            .iter()
            .any(|t| t.subject == "carol" && t.predicate == ":ancestor"));
        Ok(())
    }

    #[test]
    fn test_engine_zero_iterations() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = N3Engine::new();
        engine.assert_fact(Triple::new("a", ":p", "b"));
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":p", "?y")],
            vec![tf("?x", ":q", "?y")],
        ));
        let facts = engine.run_bounded(0)?;
        assert_eq!(facts.len(), 1);
        Ok(())
    }

    #[test]
    fn test_engine_variable_binding_consistency() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![tf("?x", ":type", "?t"), tf("?y", ":type", "?t")],
            vec![tf("?x", ":sameTypeAs", "?y")],
        ));
        engine.assert_fact(Triple::new("alice", ":type", "Human"));
        engine.assert_fact(Triple::new("bob", ":type", "Human"));
        engine.assert_fact(Triple::new("rex", ":type", "Dog"));
        let facts = engine.run()?;
        assert!(
            facts
                .iter()
                .any(|t| t.subject == "alice" && t.predicate == ":sameTypeAs" && t.object == "bob"),
            "got: {:?}",
            facts
        );
        assert!(!facts
            .iter()
            .any(|t| t.subject == "alice" && t.predicate == ":sameTypeAs" && t.object == "rex"));
        Ok(())
    }

    #[test]
    fn test_engine_three_antecedent_chain() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = N3Engine::new();
        engine.add_rule(N3Rule::new(
            vec![
                tf("?a", ":p1", "?b"),
                tf("?b", ":p2", "?c"),
                tf("?c", ":p3", "?d"),
            ],
            vec![tf("?a", ":chain", "?d")],
        ));
        engine.assert_fact(Triple::new("n1", ":p1", "n2"));
        engine.assert_fact(Triple::new("n2", ":p2", "n3"));
        engine.assert_fact(Triple::new("n3", ":p3", "n4"));
        let facts = engine.run()?;
        assert!(
            facts
                .iter()
                .any(|t| t.subject == "n1" && t.predicate == ":chain" && t.object == "n4"),
            "got: {:?}",
            facts
        );
        Ok(())
    }

    // ── Math built-ins ───────────────────────────────────────────────────────

    fn lit(v: &str) -> N3Term {
        N3Term::Literal {
            value: v.to_string(),
            datatype: None,
            lang: None,
        }
    }

    #[test]
    fn test_math_gt_passes() {
        let bi = N3BuiltIn::MathGreaterThan {
            left: lit("25"),
            right: lit("18"),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_gt_fails() {
        let bi = N3BuiltIn::MathGreaterThan {
            left: lit("10"),
            right: lit("18"),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_lt_passes() {
        let bi = N3BuiltIn::MathLessThan {
            left: lit("10"),
            right: lit("18"),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_lt_fails() {
        let bi = N3BuiltIn::MathLessThan {
            left: lit("25"),
            right: lit("18"),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_sum_binds_result() {
        let bi = N3BuiltIn::MathSum {
            args: vec![lit("3"), lit("4")],
            result: N3Term::Variable("r".to_string()),
        };
        let mut b = Bindings::new();
        assert!(N3Engine::evaluate_builtin(&bi, &mut b));
        assert!(matches!(b.get("r"), Some(N3Term::Literal { value, .. }) if value == "7"));
    }

    #[test]
    fn test_math_sum_checks_correct_result() {
        let bi = N3BuiltIn::MathSum {
            args: vec![lit("3"), lit("4")],
            result: lit("7"),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_sum_wrong_result() {
        let bi = N3BuiltIn::MathSum {
            args: vec![lit("3"), lit("4")],
            result: lit("8"),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_difference() {
        let bi = N3BuiltIn::MathDifference {
            args: vec![lit("10"), lit("3")],
            result: N3Term::Variable("r".to_string()),
        };
        let mut b = Bindings::new();
        assert!(N3Engine::evaluate_builtin(&bi, &mut b));
        assert!(matches!(b.get("r"), Some(N3Term::Literal { value, .. }) if value == "7"));
    }

    #[test]
    fn test_math_product_binds_result() {
        let bi = N3BuiltIn::MathProduct {
            args: vec![lit("3"), lit("4")],
            result: N3Term::Variable("r".to_string()),
        };
        let mut b = Bindings::new();
        assert!(N3Engine::evaluate_builtin(&bi, &mut b));
        assert!(matches!(b.get("r"), Some(N3Term::Literal { value, .. }) if value == "12"));
    }

    #[test]
    fn test_math_product_checks_result() {
        let bi = N3BuiltIn::MathProduct {
            args: vec![lit("3"), lit("4")],
            result: lit("12"),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_quotient() {
        let bi = N3BuiltIn::MathQuotient {
            args: vec![lit("10"), lit("2")],
            result: N3Term::Variable("r".to_string()),
        };
        let mut b = Bindings::new();
        assert!(N3Engine::evaluate_builtin(&bi, &mut b));
        assert!(matches!(b.get("r"), Some(N3Term::Literal { value, .. }) if value == "5"));
    }

    #[test]
    fn test_math_quotient_div_zero() {
        let bi = N3BuiltIn::MathQuotient {
            args: vec![lit("10"), lit("0")],
            result: N3Term::Variable("r".to_string()),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_equal_to_passes() {
        let bi = N3BuiltIn::MathEqualTo {
            left: lit("5"),
            right: lit("5"),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_math_equal_to_fails() {
        let bi = N3BuiltIn::MathEqualTo {
            left: lit("5"),
            right: lit("6"),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    // ── String built-ins ─────────────────────────────────────────────────────

    #[test]
    fn test_string_concat_binds() {
        let bi = N3BuiltIn::StringConcatenation {
            args: vec![lit("Hello"), lit(" World")],
            result: N3Term::Variable("r".to_string()),
        };
        let mut b = Bindings::new();
        assert!(N3Engine::evaluate_builtin(&bi, &mut b));
        assert!(
            matches!(b.get("r"), Some(N3Term::Literal { value, .. }) if value == "Hello World")
        );
    }

    #[test]
    fn test_string_concat_checks() {
        let bi = N3BuiltIn::StringConcatenation {
            args: vec![lit("foo"), lit("bar")],
            result: lit("foobar"),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_string_length_binds() {
        let bi = N3BuiltIn::StringLength {
            input: lit("hello"),
            result: N3Term::Variable("len".to_string()),
        };
        let mut b = Bindings::new();
        assert!(N3Engine::evaluate_builtin(&bi, &mut b));
        assert!(matches!(b.get("len"), Some(N3Term::Literal { value, .. }) if value == "5"));
    }

    #[test]
    fn test_string_length_empty() {
        let bi = N3BuiltIn::StringLength {
            input: lit(""),
            result: N3Term::Variable("l".to_string()),
        };
        let mut b = Bindings::new();
        assert!(N3Engine::evaluate_builtin(&bi, &mut b));
        assert!(matches!(b.get("l"), Some(N3Term::Literal { value, .. }) if value == "0"));
    }

    #[test]
    fn test_string_contains_passes() {
        let bi = N3BuiltIn::StringContains {
            subject: lit("Hello World"),
            substring: lit("World"),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_string_contains_fails() {
        let bi = N3BuiltIn::StringContains {
            subject: lit("Hello"),
            substring: lit("World"),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    // ── Log built-ins ────────────────────────────────────────────────────────

    #[test]
    fn test_log_equal_same() {
        let bi = N3BuiltIn::LogEqual {
            left: N3Term::Iri("http://x.org/a".to_string()),
            right: N3Term::Iri("http://x.org/a".to_string()),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_log_equal_different() {
        let bi = N3BuiltIn::LogEqual {
            left: N3Term::Iri("http://x.org/a".to_string()),
            right: N3Term::Iri("http://x.org/b".to_string()),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_log_not_equal_different() {
        let bi = N3BuiltIn::LogNotEqual {
            left: N3Term::Iri("http://x.org/a".to_string()),
            right: N3Term::Iri("http://x.org/b".to_string()),
        };
        assert!(N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    #[test]
    fn test_log_not_equal_same() {
        let bi = N3BuiltIn::LogNotEqual {
            left: N3Term::Iri("http://x.org/x".to_string()),
            right: N3Term::Iri("http://x.org/x".to_string()),
        };
        assert!(!N3Engine::evaluate_builtin(&bi, &mut Bindings::new()));
    }

    // ── Universal variable tests ─────────────────────────────────────────────

    #[test]
    fn test_universal_vars_collected() -> Result<(), Box<dyn std::error::Error>> {
        let n3 = "@forAll :x, :y . { :x :knows :y } => { :y :knownBy :x } .";
        let rules = N3Parser::parse(n3)?;
        assert_eq!(rules[0].universals.len(), 2);
        Ok(())
    }

    #[test]
    fn test_existential_vars_collected() -> Result<(), Box<dyn std::error::Error>> {
        let n3 = "@forSome :z . { ?x :knows ?y } => { ?x :connects :z } .";
        let rules = N3Parser::parse(n3)?;
        assert_eq!(rules[0].existentials.len(), 1);
        Ok(())
    }

    #[test]
    fn test_universal_term_in_rule_fires() -> Result<(), Box<dyn std::error::Error>> {
        let rule = N3Rule::new(
            vec![N3Formula::Triple {
                subject: N3Term::Universal("x".to_string()),
                predicate: N3Term::Iri(":classOf".to_string()),
                object: N3Term::Universal("y".to_string()),
            }],
            vec![N3Formula::Triple {
                subject: N3Term::Universal("y".to_string()),
                predicate: N3Term::Iri(":memberOf".to_string()),
                object: N3Term::Universal("x".to_string()),
            }],
        )
        .with_universals(vec!["x".to_string(), "y".to_string()]);
        let mut engine = N3Engine::new();
        engine.add_rule(rule);
        engine.assert_fact(Triple::new("Animal", ":classOf", "Dog"));
        let facts = engine.run()?;
        assert!(
            facts
                .iter()
                .any(|t| t.subject == "Dog" && t.predicate == ":memberOf" && t.object == "Animal"),
            "got: {:?}",
            facts
        );
        Ok(())
    }

    // ── N3Term display / helpers ─────────────────────────────────────────────

    #[test]
    fn test_term_display_iri() {
        assert_eq!(
            format!("{}", N3Term::Iri("http://x.org".to_string())),
            "<http://x.org>"
        );
    }

    #[test]
    fn test_term_display_variable() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(format!("{}", N3Term::Variable("x".to_string())), "?x");
        Ok(())
    }

    #[test]
    fn test_term_display_literal() {
        assert_eq!(
            format!(
                "{}",
                N3Term::Literal {
                    value: "hi".to_string(),
                    datatype: None,
                    lang: None
                }
            ),
            "\"hi\""
        );
    }

    #[test]
    fn test_term_display_blank_node() {
        assert_eq!(format!("{}", N3Term::BlankNode("b1".to_string())), "_:b1");
    }

    #[test]
    fn test_term_display_universal() {
        assert_eq!(format!("{}", N3Term::Universal("u".to_string())), "!u");
    }

    #[test]
    fn test_term_is_variable_true() {
        assert!(N3Term::Variable("v".to_string()).is_variable());
        assert!(N3Term::Universal("u".to_string()).is_variable());
    }

    #[test]
    fn test_term_is_variable_false() {
        assert!(!N3Term::Iri("x".to_string()).is_variable());
        assert!(!N3Term::BlankNode("b".to_string()).is_variable());
    }

    // ── N3Rule builder / display ─────────────────────────────────────────────

    #[test]
    fn test_rule_with_universals() {
        let r = N3Rule::new(vec![], vec![]).with_universals(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(r.universals.len(), 2);
    }

    #[test]
    fn test_rule_with_existentials() {
        let r = N3Rule::new(vec![], vec![]).with_existentials(vec!["z".to_string()]);
        assert_eq!(r.existentials.len(), 1);
    }

    #[test]
    fn test_rule_display() {
        let r = N3Rule::new(vec![], vec![]);
        assert!(format!("{}", r).contains("N3Rule"));
    }

    // ── Triple ───────────────────────────────────────────────────────────────

    #[test]
    fn test_triple_equality() {
        assert_eq!(Triple::new("a", "b", "c"), Triple::new("a", "b", "c"));
    }

    #[test]
    fn test_triple_inequality() {
        assert_ne!(Triple::new("a", "b", "c"), Triple::new("a", "b", "d"));
    }

    #[test]
    fn test_engine_no_duplicate_facts() {
        let mut engine = N3Engine::new();
        engine.assert_fact(Triple::new("a", "p", "b"));
        engine.assert_fact(Triple::new("a", "p", "b"));
        assert_eq!(engine.facts.len(), 1);
    }

    #[test]
    fn test_engine_multiple_facts_distinct() {
        let mut engine = N3Engine::new();
        engine.assert_fact(Triple::new("a", "p", "b"));
        engine.assert_fact(Triple::new("b", "p", "c"));
        engine.assert_fact(Triple::new("c", "p", "d"));
        assert_eq!(engine.facts.len(), 3);
    }
}
