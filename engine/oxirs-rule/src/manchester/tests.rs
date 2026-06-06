//! Comprehensive tests for the OWL Manchester Syntax parser and emitter.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::manchester::{emit, parse, ManchesterError, ManchesterExpr};

    // ─── Helper ───────────────────────────────────────────────────────────────

    fn class(s: &str) -> ManchesterExpr {
        ManchesterExpr::Class(s.to_string())
    }

    fn boxed(expr: ManchesterExpr) -> Box<ManchesterExpr> {
        Box::new(expr)
    }

    // ─── 1. Simple class name ─────────────────────────────────────────────────

    #[test]
    fn test_parse_simple_class() {
        let expr = parse("Person").unwrap();
        assert_eq!(expr, class("Person"));
    }

    // ─── 2. Intersection (and) ────────────────────────────────────────────────

    #[test]
    fn test_parse_and() {
        let expr = parse("A and B").unwrap();
        assert_eq!(expr, ManchesterExpr::And(vec![class("A"), class("B")]));
    }

    // ─── 3. Union (or) ───────────────────────────────────────────────────────

    #[test]
    fn test_parse_or() {
        let expr = parse("A or B").unwrap();
        assert_eq!(expr, ManchesterExpr::Or(vec![class("A"), class("B")]));
    }

    // ─── 4. Complement (not) ─────────────────────────────────────────────────

    #[test]
    fn test_parse_not() {
        let expr = parse("not A").unwrap();
        assert_eq!(expr, ManchesterExpr::Not(boxed(class("A"))));
    }

    // ─── 5. ObjectSomeValuesFrom ──────────────────────────────────────────────

    #[test]
    fn test_parse_some() {
        let expr = parse("hasChild some Person").unwrap();
        assert_eq!(
            expr,
            ManchesterExpr::Some {
                property: "hasChild".to_string(),
                filler: boxed(class("Person")),
            }
        );
    }

    // ─── 6. ObjectAllValuesFrom ───────────────────────────────────────────────

    #[test]
    fn test_parse_only() {
        let expr = parse("hasChild only Person").unwrap();
        assert_eq!(
            expr,
            ManchesterExpr::Only {
                property: "hasChild".to_string(),
                filler: boxed(class("Person")),
            }
        );
    }

    // ─── 7. ObjectMinCardinality (no filler) ─────────────────────────────────

    #[test]
    fn test_parse_min() {
        let expr = parse("hasChild min 3").unwrap();
        assert_eq!(
            expr,
            ManchesterExpr::Min {
                property: "hasChild".to_string(),
                cardinality: 3,
                filler: None,
            }
        );
    }

    // ─── 8. ObjectMinCardinality (with filler) ────────────────────────────────

    #[test]
    fn test_parse_min_filler() {
        let expr = parse("hasChild min 3 Person").unwrap();
        assert_eq!(
            expr,
            ManchesterExpr::Min {
                property: "hasChild".to_string(),
                cardinality: 3,
                filler: Some(boxed(class("Person"))),
            }
        );
    }

    // ─── 9. ObjectMaxCardinality ─────────────────────────────────────────────

    #[test]
    fn test_parse_max() {
        let expr = parse("hasChild max 2").unwrap();
        assert_eq!(
            expr,
            ManchesterExpr::Max {
                property: "hasChild".to_string(),
                cardinality: 2,
                filler: None,
            }
        );
    }

    #[test]
    fn test_parse_max_filler() {
        let expr = parse("hasChild max 2 Person").unwrap();
        assert_eq!(
            expr,
            ManchesterExpr::Max {
                property: "hasChild".to_string(),
                cardinality: 2,
                filler: Some(boxed(class("Person"))),
            }
        );
    }

    // ─── 10. ObjectExactCardinality ───────────────────────────────────────────

    #[test]
    fn test_parse_exactly() {
        let expr = parse("hasChild exactly 1").unwrap();
        assert_eq!(
            expr,
            ManchesterExpr::Exactly {
                property: "hasChild".to_string(),
                cardinality: 1,
                filler: None,
            }
        );
    }

    #[test]
    fn test_parse_exactly_filler() {
        let expr = parse("hasChild exactly 1 Person").unwrap();
        assert_eq!(
            expr,
            ManchesterExpr::Exactly {
                property: "hasChild".to_string(),
                cardinality: 1,
                filler: Some(boxed(class("Person"))),
            }
        );
    }

    // ─── 11. ObjectHasValue ───────────────────────────────────────────────────

    #[test]
    fn test_parse_has_value() {
        let expr = parse("knows value JohnDoe").unwrap();
        assert_eq!(
            expr,
            ManchesterExpr::HasValue {
                property: "knows".to_string(),
                individual: "JohnDoe".to_string(),
            }
        );
    }

    // ─── 12. ObjectOneOf ──────────────────────────────────────────────────────

    #[test]
    fn test_parse_one_of() {
        let expr = parse("{Alice Bob}").unwrap();
        assert_eq!(
            expr,
            ManchesterExpr::OneOf(vec!["Alice".to_string(), "Bob".to_string()])
        );
    }

    // ─── 13. Nested and / or ──────────────────────────────────────────────────

    #[test]
    fn test_parse_nested_and_or() {
        let expr = parse("A and (B or C)").unwrap();
        assert_eq!(
            expr,
            ManchesterExpr::And(vec![
                class("A"),
                ManchesterExpr::Or(vec![class("B"), class("C")]),
            ])
        );
    }

    // ─── 14. Emit round-trip: simple ─────────────────────────────────────────

    #[test]
    fn test_emit_roundtrip_simple() {
        let original = parse("Person").unwrap();
        let text = emit(&original).unwrap();
        let reparsed = parse(&text).unwrap();
        assert_eq!(original, reparsed);
    }

    // ─── 15. Emit round-trip: complex ────────────────────────────────────────

    #[test]
    fn test_emit_roundtrip_complex() {
        // A complex expression exercising most node types
        let input = "hasChild some (Person and (not Animal))";
        let original = parse(input).unwrap();
        let text = emit(&original).unwrap();
        let reparsed = parse(&text).unwrap();
        assert_eq!(original, reparsed, "round-trip failed: emitted `{text}`");
    }

    // ─── 16. Parse error: empty input ────────────────────────────────────────

    #[test]
    fn test_parse_error_empty() {
        let result = parse("");
        assert!(
            result.is_err(),
            "expected Err for empty string, got Ok({result:?})"
        );
        match result {
            Err(ManchesterError::ParseError { .. }) => {}
            Err(other) => panic!("expected ParseError, got {other:?}"),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }

    // ─── 17. Prefixed names (e.g. owl:Thing) ─────────────────────────────────

    #[test]
    fn test_parse_prefixed_name() {
        let expr = parse("owl:Thing").unwrap();
        assert_eq!(expr, class("owl:Thing"));
    }

    // ─── 18. Three-way `and` ─────────────────────────────────────────────────

    #[test]
    fn test_three_way_and() {
        let expr = parse("A and B and C").unwrap();
        assert_eq!(
            expr,
            ManchesterExpr::And(vec![class("A"), class("B"), class("C")])
        );
    }

    // ─── Additional: emit specific forms ─────────────────────────────────────

    #[test]
    fn test_emit_some() {
        let expr = ManchesterExpr::Some {
            property: "hasChild".to_string(),
            filler: boxed(class("Person")),
        };
        assert_eq!(emit(&expr).unwrap(), "hasChild some Person");
    }

    #[test]
    fn test_emit_only() {
        let expr = ManchesterExpr::Only {
            property: "hasChild".to_string(),
            filler: boxed(class("Person")),
        };
        assert_eq!(emit(&expr).unwrap(), "hasChild only Person");
    }

    #[test]
    fn test_emit_min_no_filler() {
        let expr = ManchesterExpr::Min {
            property: "hasChild".to_string(),
            cardinality: 3,
            filler: None,
        };
        assert_eq!(emit(&expr).unwrap(), "hasChild min 3");
    }

    #[test]
    fn test_emit_min_with_filler() {
        let expr = ManchesterExpr::Min {
            property: "hasChild".to_string(),
            cardinality: 3,
            filler: Some(boxed(class("Person"))),
        };
        assert_eq!(emit(&expr).unwrap(), "hasChild min 3 Person");
    }

    #[test]
    fn test_emit_max() {
        let expr = ManchesterExpr::Max {
            property: "hasChild".to_string(),
            cardinality: 2,
            filler: None,
        };
        assert_eq!(emit(&expr).unwrap(), "hasChild max 2");
    }

    #[test]
    fn test_emit_exactly() {
        let expr = ManchesterExpr::Exactly {
            property: "hasChild".to_string(),
            cardinality: 1,
            filler: None,
        };
        assert_eq!(emit(&expr).unwrap(), "hasChild exactly 1");
    }

    #[test]
    fn test_emit_has_value() {
        let expr = ManchesterExpr::HasValue {
            property: "knows".to_string(),
            individual: "JohnDoe".to_string(),
        };
        assert_eq!(emit(&expr).unwrap(), "knows value JohnDoe");
    }

    #[test]
    fn test_emit_one_of() {
        let expr = ManchesterExpr::OneOf(vec!["Alice".to_string(), "Bob".to_string()]);
        assert_eq!(emit(&expr).unwrap(), "{Alice Bob}");
    }

    #[test]
    fn test_emit_not() {
        let expr = ManchesterExpr::Not(boxed(class("Animal")));
        assert_eq!(emit(&expr).unwrap(), "not Animal");
    }

    #[test]
    fn test_emit_and_top_level() {
        let expr = ManchesterExpr::And(vec![class("A"), class("B")]);
        // Top-level: no outer parens
        assert_eq!(emit(&expr).unwrap(), "A and B");
    }

    #[test]
    fn test_emit_or_nested_gets_parens() {
        // When Or is nested inside And it must be parenthesized
        let expr = ManchesterExpr::And(vec![
            class("A"),
            ManchesterExpr::Or(vec![class("B"), class("C")]),
        ]);
        let text = emit(&expr).unwrap();
        assert_eq!(text, "A and (B or C)");
    }

    #[test]
    fn test_parse_error_unexpected_token() {
        // A lone closing paren should produce a parse error
        let result = parse(")");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_single_one_of() {
        let expr = parse("{Alice}").unwrap();
        assert_eq!(expr, ManchesterExpr::OneOf(vec!["Alice".to_string()]));
    }

    #[test]
    fn test_parse_nested_not_inside_or() {
        let expr = parse("not A or B").unwrap();
        // Grammar: or_expr -> and_expr ('or' and_expr)*
        //          and_expr -> not_expr ('and' not_expr)*
        //          not_expr -> 'not' primary | primary
        // So `not A or B` = Or([Not(A), B])
        assert_eq!(
            expr,
            ManchesterExpr::Or(vec![ManchesterExpr::Not(boxed(class("A"))), class("B"),])
        );
    }

    #[test]
    fn test_roundtrip_cardinality_with_filler() {
        let input = "hasChild exactly 2 Person";
        let original = parse(input).unwrap();
        let emitted = emit(&original).unwrap();
        let reparsed = parse(&emitted).unwrap();
        assert_eq!(original, reparsed);
        assert_eq!(emitted, "hasChild exactly 2 Person");
    }

    #[test]
    fn test_roundtrip_has_value() {
        let input = "knows value JohnDoe";
        let original = parse(input).unwrap();
        let emitted = emit(&original).unwrap();
        let reparsed = parse(&emitted).unwrap();
        assert_eq!(original, reparsed);
    }

    #[test]
    fn test_roundtrip_one_of() {
        let input = "{Alice Bob Charlie}";
        let original = parse(input).unwrap();
        let emitted = emit(&original).unwrap();
        let reparsed = parse(&emitted).unwrap();
        assert_eq!(original, reparsed);
    }

    #[test]
    fn test_parse_complex_nested() {
        // (hasChild some Person) and (knows value JohnDoe)
        let input = "(hasChild some Person) and (knows value JohnDoe)";
        let expr = parse(input).unwrap();
        assert_eq!(
            expr,
            ManchesterExpr::And(vec![
                ManchesterExpr::Some {
                    property: "hasChild".to_string(),
                    filler: boxed(class("Person")),
                },
                ManchesterExpr::HasValue {
                    property: "knows".to_string(),
                    individual: "JohnDoe".to_string(),
                },
            ])
        );
    }

    #[test]
    fn test_parse_prefixed_name_in_restriction() {
        let expr = parse("owl:ObjectProperty some owl:Thing").unwrap();
        assert_eq!(
            expr,
            ManchesterExpr::Some {
                property: "owl:ObjectProperty".to_string(),
                filler: boxed(class("owl:Thing")),
            }
        );
    }
}
