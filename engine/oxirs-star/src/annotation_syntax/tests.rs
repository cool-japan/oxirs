//! Tests for the annotation_syntax module
#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::annotation_syntax::{
        expander::{expand_annotations, to_explicit_turtle},
        tokenizer::{find_annotation_blocks, tokenize_annotation_block, AnnotationToken},
        AnnotatedTriple, AnnotationLiteral, AnnotationPair, AnnotationParser,
        AnnotationSyntaxError, AnnotationValue, RdfStarTriple, StarTerm,
    };

    // ---- helper ----
    fn named(iri: &str) -> StarTerm {
        StarTerm::NamedNode(iri.to_string())
    }

    fn ann_named(iri: &str) -> AnnotationValue {
        AnnotationValue::NamedNode(iri.to_string())
    }

    fn base_triple() -> RdfStarTriple {
        RdfStarTriple::new(
            named("http://example.org/s"),
            named("http://example.org/p"),
            named("http://example.org/o"),
        )
    }

    // ---- parse tests ----

    #[test]
    fn test_parse_simple_annotation() {
        let parser = AnnotationParser::new();
        let input = "<http://example.org/s> <http://example.org/p> <http://example.org/o> \
                     {| <http://example.org/ap> <http://example.org/ao> |}";
        let result = parser.parse_annotated_triple(input);
        assert!(result.is_ok(), "parse failed: {:?}", result.err());
        let annotated = result.unwrap();
        assert_eq!(annotated.annotations.len(), 1);
        assert_eq!(annotated.annotations[0].predicate, "http://example.org/ap");
        assert_eq!(
            annotated.annotations[0].object,
            ann_named("http://example.org/ao")
        );
    }

    #[test]
    fn test_parse_multiple_annotations() {
        let parser = AnnotationParser::new();
        let input =
            "<http://example.org/s> <http://example.org/p> <http://example.org/o> \
             {| <http://example.org/ap1> <http://example.org/ao1> ; <http://example.org/ap2> <http://example.org/ao2> |}";
        let result = parser.parse_annotated_triple(input);
        assert!(result.is_ok(), "parse failed: {:?}", result.err());
        let annotated = result.unwrap();
        assert_eq!(annotated.annotations.len(), 2);
        assert_eq!(annotated.annotations[0].predicate, "http://example.org/ap1");
        assert_eq!(annotated.annotations[1].predicate, "http://example.org/ap2");
    }

    #[test]
    fn test_parse_annotation_with_literal() {
        let parser = AnnotationParser::new();
        let input = "<http://example.org/s> <http://example.org/p> <http://example.org/o> \
             {| <http://example.org/certainty> \"0.95\" |}";
        let result = parser.parse_annotated_triple(input);
        assert!(result.is_ok(), "parse failed: {:?}", result.err());
        let annotated = result.unwrap();
        assert_eq!(annotated.annotations.len(), 1);
        if let AnnotationValue::Literal(lit) = &annotated.annotations[0].object {
            assert_eq!(lit.value, "0.95");
            assert!(lit.language.is_none());
            assert!(lit.datatype.is_none());
        } else {
            panic!("expected literal annotation object");
        }
    }

    #[test]
    fn test_parse_annotation_with_typed_literal() {
        let parser = AnnotationParser::new();
        let input = "<http://example.org/s> <http://example.org/p> <http://example.org/o> \
             {| <http://example.org/score> \"42\"^^<http://www.w3.org/2001/XMLSchema#integer> |}";
        let result = parser.parse_annotated_triple(input);
        assert!(result.is_ok(), "parse failed: {:?}", result.err());
        let annotated = result.unwrap();
        assert_eq!(annotated.annotations.len(), 1);
        if let AnnotationValue::Literal(lit) = &annotated.annotations[0].object {
            assert_eq!(lit.value, "42");
            assert_eq!(
                lit.datatype.as_deref(),
                Some("http://www.w3.org/2001/XMLSchema#integer")
            );
        } else {
            panic!("expected typed literal annotation object");
        }
    }

    #[test]
    fn test_parse_annotation_with_blank_node() {
        let parser = AnnotationParser::new();
        let input = "<http://example.org/s> <http://example.org/p> <http://example.org/o> \
                     {| <http://example.org/ap> _:b0 |}";
        let result = parser.parse_annotated_triple(input);
        assert!(result.is_ok(), "parse failed: {:?}", result.err());
        let annotated = result.unwrap();
        assert_eq!(annotated.annotations.len(), 1);
        assert_eq!(
            annotated.annotations[0].object,
            AnnotationValue::BlankNode("b0".to_string())
        );
    }

    // ---- expand tests ----

    #[test]
    fn test_expand_single_annotation() {
        let base = base_triple();
        let annotations = vec![AnnotationPair {
            predicate: "http://example.org/certainty".to_string(),
            object: ann_named("http://example.org/high"),
        }];
        let annotated = AnnotatedTriple::new(base, annotations);
        let expanded = expand_annotations(&annotated);
        assert_eq!(expanded.len(), 1);

        // First element should be a QuotedTriple of the base
        if let StarTerm::QuotedTriple(qt) = &expanded[0].0 {
            assert_eq!(qt.subject, named("http://example.org/s"));
        } else {
            panic!("expected QuotedTriple subject");
        }

        assert_eq!(expanded[0].1, "http://example.org/certainty");
    }

    #[test]
    fn test_expand_multiple_annotations() {
        let base = base_triple();
        let annotations = vec![
            AnnotationPair {
                predicate: "http://example.org/ap1".to_string(),
                object: ann_named("http://example.org/ao1"),
            },
            AnnotationPair {
                predicate: "http://example.org/ap2".to_string(),
                object: ann_named("http://example.org/ao2"),
            },
            AnnotationPair {
                predicate: "http://example.org/ap3".to_string(),
                object: ann_named("http://example.org/ao3"),
            },
        ];
        let annotated = AnnotatedTriple::new(base, annotations);
        let expanded = expand_annotations(&annotated);
        assert_eq!(expanded.len(), 3);

        // All should share the same quoted triple subject
        for (qt_term, _, _) in &expanded {
            assert!(matches!(qt_term, StarTerm::QuotedTriple(_)));
        }
    }

    #[test]
    fn test_to_explicit_turtle() {
        let base = base_triple();
        let annotations = vec![AnnotationPair {
            predicate: "http://example.org/certainty".to_string(),
            object: ann_named("http://example.org/high"),
        }];
        let annotated = AnnotatedTriple::new(base, annotations);
        let turtle = to_explicit_turtle(&annotated);

        // Should contain the quoted form
        assert!(turtle.contains("<< "), "missing quoted triple: {}", turtle);
        assert!(turtle.contains(" >>"), "missing closing >>: {}", turtle);
        assert!(
            turtle.contains("http://example.org/certainty"),
            "missing annotation predicate: {}",
            turtle
        );
        assert!(
            turtle.contains("http://example.org/high"),
            "missing annotation object: {}",
            turtle
        );
    }

    // ---- tokenizer tests ----

    #[test]
    fn test_tokenize_annotation_block() {
        let input = "<http://example.org/p> <http://example.org/o>";
        let tokens = tokenize_annotation_block(input).unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(
            tokens[0],
            AnnotationToken::NamedNode("http://example.org/p".to_string())
        );
        assert_eq!(
            tokens[1],
            AnnotationToken::NamedNode("http://example.org/o".to_string())
        );
    }

    #[test]
    fn test_find_annotation_blocks() {
        let input = "<s> <p> <o> {| <ap1> <ao1> |} .\n<s2> <p2> <o2> {| <ap2> <ao2> |} .";
        let blocks = find_annotation_blocks(input);
        assert_eq!(
            blocks.len(),
            2,
            "expected 2 annotation blocks, got {}",
            blocks.len()
        );

        // First block
        let (s1, e1) = blocks[0];
        let block1 = &input[s1..e1];
        assert!(block1.contains("ap1"), "first block: {}", block1);

        // Second block
        let (s2, e2) = blocks[1];
        let block2 = &input[s2..e2];
        assert!(block2.contains("ap2"), "second block: {}", block2);
    }

    #[test]
    fn test_nested_quoted_triple_subject() {
        // A triple whose subject is itself a quoted triple
        let inner = RdfStarTriple::new(
            named("http://example.org/s"),
            named("http://example.org/p"),
            named("http://example.org/o"),
        );
        let quoted_subject = StarTerm::QuotedTriple(Box::new(inner));
        let outer = RdfStarTriple::new(
            quoted_subject,
            named("http://example.org/certainty"),
            named("http://example.org/high"),
        );

        let turtle = outer.to_turtle();
        assert!(turtle.contains("<< "), "missing << in: {}", turtle);
        assert!(turtle.contains(" >>"), "missing >> in: {}", turtle);
    }

    #[test]
    fn test_error_unclosed_annotation() {
        let parser = AnnotationParser::new();
        let input = "<http://example.org/s> <http://example.org/p> <http://example.org/o> \
                     {| <http://example.org/p>";
        let result = parser.parse_annotated_triple(input);
        assert!(
            matches!(result, Err(AnnotationSyntaxError::UnclosedAnnotation)),
            "expected UnclosedAnnotation, got {:?}",
            result
        );
    }

    #[test]
    fn test_error_empty_annotation() {
        let input = ""; // empty — tokenizer returns empty vec, parser raises EmptyAnnotation
        let tokens = tokenize_annotation_block(input).unwrap();
        assert!(tokens.is_empty());

        // Manually invoke the parse with an AnnotatedTriple having no annotations
        // via the block that calls parse_annotation_pairs([])
        let parser = AnnotationParser::new();
        let full_input =
            "<http://example.org/s> <http://example.org/p> <http://example.org/o> {| |}";
        let result = parser.parse_annotated_triple(full_input);
        assert!(
            matches!(result, Err(AnnotationSyntaxError::EmptyAnnotation)),
            "expected EmptyAnnotation, got {:?}",
            result
        );
    }

    #[test]
    fn test_error_invalid_predicate() {
        // Blank nodes cannot be predicates
        let input = "<http://example.org/p> _:b0 <http://example.org/o>";
        let _tokens = tokenize_annotation_block(input).unwrap();
        // tokens: NamedNode(p), BlankNode(b0), NamedNode(o)
        // The parser expects NamedNode as predicate — when it encounters a blank node
        // as "predicate" and then tries to parse the rest as object, the InvalidPredicate
        // error should surface. We test via the full parser.
        let parser = AnnotationParser::new();
        let full_input = "<http://example.org/s> <http://example.org/p> <http://example.org/o> \
             {| _:b0 <http://example.org/ao> |}";
        let result = parser.parse_annotated_triple(full_input);
        assert!(
            matches!(result, Err(AnnotationSyntaxError::InvalidPredicate(_))),
            "expected InvalidPredicate, got {:?}",
            result
        );
    }

    #[test]
    fn test_roundtrip_explicit() {
        let parser = AnnotationParser::new();
        let input = "<http://example.org/s> <http://example.org/p> <http://example.org/o> \
             {| <http://example.org/certainty> \"high\" |}";
        let annotated = parser.parse_annotated_triple(input).unwrap();

        // Serialize to explicit form
        let explicit = to_explicit_turtle(&annotated);

        // Must contain the base triple and the annotation
        assert!(
            explicit.contains("http://example.org/s"),
            "missing subject: {}",
            explicit
        );
        assert!(
            explicit.contains("http://example.org/certainty"),
            "missing annotation predicate: {}",
            explicit
        );
        assert!(
            explicit.contains("high"),
            "missing annotation value: {}",
            explicit
        );
        // Must contain quoted triple form
        assert!(
            explicit.contains("<<"),
            "missing quoted triple open: {}",
            explicit
        );
        assert!(
            explicit.contains(">>"),
            "missing quoted triple close: {}",
            explicit
        );
    }

    #[test]
    fn test_annotation_literal_with_language() {
        let lit = AnnotationLiteral {
            value: "hello".to_string(),
            language: Some("en".to_string()),
            datatype: None,
        };
        let val = AnnotationValue::Literal(lit);
        let turtle = val.to_turtle();
        assert_eq!(turtle, "\"hello\"@en");
    }

    #[test]
    fn test_annotation_value_named_node_turtle() {
        let val = ann_named("http://example.org/x");
        assert_eq!(val.to_turtle(), "<http://example.org/x>");
    }

    #[test]
    fn test_tokenize_with_typed_literal() {
        let input = "<http://example.org/p> \"42\"^^<http://www.w3.org/2001/XMLSchema#integer>";
        let tokens = tokenize_annotation_block(input).unwrap();
        assert_eq!(tokens.len(), 2);
        if let AnnotationToken::Literal(val, lang, dt) = &tokens[1] {
            assert_eq!(val, "42");
            assert!(lang.is_none());
            assert_eq!(
                dt.as_deref(),
                Some("http://www.w3.org/2001/XMLSchema#integer")
            );
        } else {
            panic!("expected Literal token, got {:?}", tokens[1]);
        }
    }
}
