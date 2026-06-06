//! Tests for the N-Quads streaming parser.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::nquads_streaming::parser::parse_line;
    use crate::nquads_streaming::{
        NQuadsParseError, NQuadsStreamingParser, StreamedLiteral, StreamedQuad, StreamedTerm,
    };
    use std::io::Cursor;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn named(iri: &str) -> StreamedTerm {
        StreamedTerm::NamedNode(iri.to_string())
    }

    fn blank(label: &str) -> StreamedTerm {
        StreamedTerm::BlankNode(label.to_string())
    }

    fn plain_literal(value: &str) -> StreamedTerm {
        StreamedTerm::Literal(StreamedLiteral {
            value: value.to_string(),
            datatype: None,
            language: None,
        })
    }

    fn lang_literal(value: &str, lang: &str) -> StreamedTerm {
        StreamedTerm::Literal(StreamedLiteral {
            value: value.to_string(),
            datatype: None,
            language: Some(lang.to_string()),
        })
    }

    fn typed_literal(value: &str, datatype: &str) -> StreamedTerm {
        StreamedTerm::Literal(StreamedLiteral {
            value: value.to_string(),
            datatype: Some(datatype.to_string()),
            language: None,
        })
    }

    // ── 1: simple triple ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_simple_triple() {
        let quad = parse_line("<http://s> <http://p> <http://o> .", 1)
            .expect("no error")
            .expect("not blank");
        assert_eq!(quad.subject, named("http://s"));
        assert_eq!(quad.predicate, named("http://p"));
        assert_eq!(quad.object, named("http://o"));
        assert!(quad.graph_name.is_none());
    }

    // ── 2: quad with graph name ───────────────────────────────────────────────

    #[test]
    fn test_parse_quad() {
        let quad = parse_line("<http://s> <http://p> <http://o> <http://g> .", 1)
            .expect("no error")
            .expect("not blank");
        assert_eq!(quad.subject, named("http://s"));
        assert_eq!(quad.predicate, named("http://p"));
        assert_eq!(quad.object, named("http://o"));
        assert_eq!(quad.graph_name, Some(named("http://g")));
    }

    // ── 3: blank node subject ─────────────────────────────────────────────────

    #[test]
    fn test_parse_blank_node_subject() {
        let quad = parse_line("_:b0 <http://p> <http://o> .", 1)
            .expect("no error")
            .expect("not blank");
        assert_eq!(quad.subject, blank("b0"));
        assert_eq!(quad.predicate, named("http://p"));
        assert_eq!(quad.object, named("http://o"));
    }

    // ── 4: blank node object ──────────────────────────────────────────────────

    #[test]
    fn test_parse_blank_node_object() {
        let quad = parse_line("<http://s> <http://p> _:b0 .", 1)
            .expect("no error")
            .expect("not blank");
        assert_eq!(quad.subject, named("http://s"));
        assert_eq!(quad.object, blank("b0"));
    }

    // ── 5: plain string literal ───────────────────────────────────────────────

    #[test]
    fn test_parse_string_literal() {
        let quad = parse_line(r#"<http://s> <http://p> "hello" ."#, 1)
            .expect("no error")
            .expect("not blank");
        assert_eq!(quad.object, plain_literal("hello"));
    }

    // ── 6: language-tagged literal ────────────────────────────────────────────

    #[test]
    fn test_parse_lang_literal() {
        let quad = parse_line(r#"<http://s> <http://p> "bonjour"@fr ."#, 1)
            .expect("no error")
            .expect("not blank");
        assert_eq!(quad.object, lang_literal("bonjour", "fr"));
    }

    // ── 7: typed literal ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_typed_literal() {
        let quad = parse_line(
            r#"<http://s> <http://p> "42"^^<http://www.w3.org/2001/XMLSchema#integer> ."#,
            1,
        )
        .expect("no error")
        .expect("not blank");
        assert_eq!(
            quad.object,
            typed_literal("42", "http://www.w3.org/2001/XMLSchema#integer")
        );
    }

    // ── 8: escape sequences ───────────────────────────────────────────────────

    #[test]
    fn test_parse_escape_sequences() {
        let quad = parse_line(r#"<http://s> <http://p> "hello\nworld" ."#, 1)
            .expect("no error")
            .expect("not blank");
        match &quad.object {
            StreamedTerm::Literal(lit) => {
                assert_eq!(lit.value, "hello\nworld");
            }
            _ => panic!("expected literal"),
        }
    }

    // ── 9: Unicode escape \uXXXX ──────────────────────────────────────────────

    #[test]
    fn test_parse_unicode_escape() {
        // A = 'A'
        let quad = parse_line(r#"<http://s> <http://p> "A" ."#, 1)
            .expect("no error")
            .expect("not blank");
        match &quad.object {
            StreamedTerm::Literal(lit) => {
                assert_eq!(lit.value, "A");
            }
            _ => panic!("expected literal"),
        }
    }

    // ── 10: comment line returns None ─────────────────────────────────────────

    #[test]
    fn test_parse_comment_line() {
        let result = parse_line("# this is a comment", 1).expect("no error");
        assert!(result.is_none(), "comment line should return None");
    }

    // ── 11: blank line returns None ───────────────────────────────────────────

    #[test]
    fn test_parse_blank_line() {
        let result = parse_line("", 1).expect("no error");
        assert!(result.is_none(), "blank line should return None");

        let result2 = parse_line("   ", 2).expect("no error");
        assert!(result2.is_none(), "whitespace-only line should return None");
    }

    // ── 12: multiple lines via iterator ───────────────────────────────────────

    #[test]
    fn test_parse_multiple_lines_iterator() {
        let data: String = (0..5)
            .map(|i| {
                format!(
                    "<http://s{}> <http://p> <http://o{}> <http://g{}> .\n",
                    i, i, i
                )
            })
            .collect();

        let parser = NQuadsStreamingParser::new(Cursor::new(data.as_bytes()));
        let quads: Vec<StreamedQuad> = parser.map(|r| r.expect("valid quad")).collect();
        assert_eq!(quads.len(), 5);

        for (i, quad) in quads.iter().enumerate() {
            assert_eq!(quad.subject, named(&format!("http://s{}", i)));
            assert_eq!(quad.predicate, named("http://p"));
            assert_eq!(quad.object, named(&format!("http://o{}", i)));
            assert_eq!(quad.graph_name, Some(named(&format!("http://g{}", i))));
        }
    }

    // ── 13: error – missing dot ───────────────────────────────────────────────

    #[test]
    fn test_parse_error_missing_dot() {
        let result = parse_line("<http://s> <http://p> <http://o>", 1);
        assert!(result.is_err(), "missing '.' should be an error");
        match result.unwrap_err() {
            NQuadsParseError::InvalidLine { .. } => {}
            e => panic!("expected InvalidLine, got {:?}", e),
        }
    }

    // ── 14: error – literal as predicate ─────────────────────────────────────

    #[test]
    fn test_parse_error_literal_as_predicate() {
        let result = parse_line(r#"<http://s> "foo" <http://o> ."#, 1);
        assert!(result.is_err(), "literal predicate should be an error");
        match result.unwrap_err() {
            NQuadsParseError::InvalidLine { .. } => {}
            e => panic!("expected InvalidLine, got {:?}", e),
        }
    }

    // ── 15: error – invalid IRI (unclosed) ────────────────────────────────────

    #[test]
    fn test_parse_error_invalid_iri() {
        let result = parse_line("<unclosed <http://p> <http://o> .", 1);
        assert!(result.is_err(), "unclosed IRI should be an error");
    }

    // ── 16: streaming large input ─────────────────────────────────────────────

    #[test]
    fn test_streaming_large_input() {
        let data: String = (0..100)
            .map(|i| {
                format!(
                    "<http://example.org/s{}> <http://example.org/p> <http://example.org/o{}> .\n",
                    i, i
                )
            })
            .collect();

        let parser = NQuadsStreamingParser::new(Cursor::new(data.as_bytes()));
        let count = parser.filter(|r| r.is_ok()).count();
        assert_eq!(count, 100);
    }

    // ── 17: blank node graph name ─────────────────────────────────────────────

    #[test]
    fn test_blank_node_graph() {
        let quad = parse_line("<http://s> <http://p> <http://o> _:g1 .", 1)
            .expect("no error")
            .expect("not blank");
        assert_eq!(quad.graph_name, Some(blank("g1")));
    }

    // ── 18: long literal ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_long_literal() {
        let long_value: String = "x".repeat(500);
        let line = format!(r#"<http://s> <http://p> "{}" ."#, long_value);
        let quad = parse_line(&line, 1).expect("no error").expect("not blank");
        match &quad.object {
            StreamedTerm::Literal(lit) => {
                assert_eq!(lit.value.len(), 500);
                assert_eq!(lit.value, long_value);
            }
            _ => panic!("expected literal"),
        }
    }
}
