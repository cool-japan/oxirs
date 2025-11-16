//! Property-based tests for N-Triples parser using proptest

use oxirs_ttl::ntriples::{NTriplesParser, NTriplesSerializer};
use oxirs_ttl::{Parser, Serializer};
use proptest::prelude::*;
use std::io::Cursor;

/// Generate valid IRI strings
fn valid_iri_strategy() -> impl Strategy<Value = String> {
    // Use a more restrictive pattern that only includes unreserved characters
    // unreserved = ALPHA / DIGIT / "-" / "." / "_" / "~"
    prop::string::string_regex("[a-zA-Z0-9._~-]+")
        .unwrap()
        .prop_map(|s| {
            // Generate a valid IRI with safe characters only
            if s.is_empty() {
                "http://example.org/resource".to_string()
            } else {
                format!("http://example.org/{}", s)
            }
        })
}

/// Generate valid literal strings (no control characters, proper escaping)
fn valid_literal_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-zA-Z0-9 .,!?;:'-]+").unwrap()
}

/// Generate valid blank node labels
/// Parser requires: [a-zA-Z_][a-zA-Z0-9_.-]* (must start with letter or underscore)
fn valid_blank_node_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-zA-Z_][a-zA-Z0-9_-]*")
        .unwrap()
        .prop_map(|s| format!("_:{}", s))
}

/// Generate valid language tags
fn valid_language_tag_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-z]{2}(-[A-Z]{2})?").unwrap()
}

#[cfg(test)]
mod ntriples_properties {
    use super::*;

    #[test]
    fn test_specific_blank_node_case() {
        // Test blank node with single space literal
        let nt = "_:a0 <http://example.org/_> \" \" .";
        let parser = NTriplesParser::new();
        let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

        match &result {
            Ok(triples) => {
                println!("Parsed successfully: {} triples", triples.len());
                assert_eq!(triples.len(), 1);
            }
            Err(e) => {
                panic!("Failed to parse: {:?}", e);
            }
        }
    }

    proptest! {
        #[test]
        fn test_simple_triple_roundtrip(
            subject_part in valid_iri_strategy(),
            predicate_part in valid_iri_strategy(),
            object_part in valid_literal_strategy()
        ) {
            let nt = format!(
                "<{}> <{}> \"{}\" .",
                subject_part, predicate_part, object_part
            );

            let parser = NTriplesParser::new();
            let nt_clone = nt.clone();
            let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt_clone)).collect();

            // Should parse successfully
            prop_assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

            let triples = result.unwrap();
            prop_assert_eq!(triples.len(), 1, "Expected exactly one triple");

            // Serialize and re-parse
            let mut buffer = Vec::new();
            let serializer = NTriplesSerializer::new();
            serializer.serialize(&triples, &mut buffer).unwrap();

            let serialized = String::from_utf8(buffer).unwrap();
            let serialized_clone = serialized.clone();
            let parser2 = NTriplesParser::new();
            let result2: Result<Vec<_>, _> = parser2.for_reader(Cursor::new(serialized_clone)).collect();

            prop_assert!(result2.is_ok(), "Failed to parse serialized output");
            let triples2 = result2.unwrap();
            prop_assert_eq!(triples.len(), triples2.len(), "Roundtrip changed triple count");
        }

        #[test]
        fn test_blank_node_subject(
            blank_label in valid_blank_node_strategy(),
            predicate_part in valid_iri_strategy(),
            object_part in valid_literal_strategy()
        ) {
            let nt = format!(
                "{} <{}> \"{}\" .",
                blank_label, predicate_part, object_part
            );

            let parser = NTriplesParser::new();
            let nt_clone = nt.clone();
            let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt_clone)).collect();

            prop_assert!(result.is_ok(), "Failed to parse blank node subject");
            let triples = result.unwrap();
            prop_assert_eq!(triples.len(), 1);
        }

        #[test]
        fn test_language_tagged_literal(
            subject_part in valid_iri_strategy(),
            predicate_part in valid_iri_strategy(),
            literal_value in valid_literal_strategy(),
            lang_tag in valid_language_tag_strategy()
        ) {
            let nt = format!(
                "<{}> <{}> \"{}\"@{} .",
                subject_part, predicate_part, literal_value, lang_tag
            );

            let parser = NTriplesParser::new();
            let nt_clone = nt.clone();
            let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt_clone)).collect();

            prop_assert!(result.is_ok(), "Failed to parse language-tagged literal");
            let triples = result.unwrap();
            prop_assert_eq!(triples.len(), 1);
        }

        #[test]
        fn test_multiple_triples(
            count in 1..10usize,
            subject_parts in prop::collection::vec(valid_iri_strategy(), 1..10),
            predicate_parts in prop::collection::vec(valid_iri_strategy(), 1..10),
            object_parts in prop::collection::vec(valid_literal_strategy(), 1..10)
        ) {
            let mut nt = String::new();
            let actual_count = count.min(subject_parts.len()).min(predicate_parts.len()).min(object_parts.len());

            for i in 0..actual_count {
                nt.push_str(&format!(
                    "<{}> <{}> \"{}\" .\n",
                    subject_parts[i], predicate_parts[i], object_parts[i]
                ));
            }

            let parser = NTriplesParser::new();
            let nt_clone = nt.clone();
            let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt_clone)).collect();

            prop_assert!(result.is_ok(), "Failed to parse multiple triples");
            let triples = result.unwrap();
            prop_assert_eq!(triples.len(), actual_count, "Parsed triple count mismatch");
        }

        #[test]
        fn test_empty_lines_and_whitespace(
            subject_part in valid_iri_strategy(),
            predicate_part in valid_iri_strategy(),
            object_part in valid_literal_strategy(),
            leading_spaces in 0..10usize,
            trailing_spaces in 0..10usize
        ) {
            let nt = format!(
                "{}<{}> <{}> \"{}\" .{}\n",
                " ".repeat(leading_spaces),
                subject_part,
                predicate_part,
                object_part,
                " ".repeat(trailing_spaces)
            );

            let parser = NTriplesParser::new();
            let nt_clone = nt.clone();
            let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt_clone)).collect();

            prop_assert!(result.is_ok(), "Failed to parse with whitespace");
            let triples = result.unwrap();
            prop_assert_eq!(triples.len(), 1);
        }

        #[test]
        fn test_literal_with_common_escapes(
            subject_part in valid_iri_strategy(),
            predicate_part in valid_iri_strategy()
        ) {
            // Test common escape sequences
            let test_cases = vec![
                "line1\\nline2",
                "tab\\there",
                "quote\\\"here",
                "backslash\\\\here",
            ];

            for literal in test_cases {
                let nt = format!(
                    "<{}> <{}> \"{}\" .",
                    subject_part, predicate_part, literal
                );

                let parser = NTriplesParser::new();
                let nt_clone = nt.clone();
                let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt_clone)).collect();

                prop_assert!(result.is_ok(), "Failed to parse escape sequences in: {}", literal);
            }
        }
    }
}

#[cfg(test)]
mod ntriples_error_cases {
    use super::*;

    proptest! {
        #[test]
        fn test_missing_period_fails(
            subject_part in valid_iri_strategy(),
            predicate_part in valid_iri_strategy(),
            object_part in valid_literal_strategy()
        ) {
            let nt = format!(
                "<{}> <{}> \"{}\"",
                subject_part, predicate_part, object_part
            );

            let parser = NTriplesParser::new();
            let nt_clone = nt.clone();
            let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt_clone)).collect();

            // Should fail due to missing period
            prop_assert!(result.is_err(), "Expected error for missing period");
        }

        #[test]
        fn test_invalid_iri_with_spaces_fails(
            predicate_part in valid_iri_strategy(),
            object_part in valid_literal_strategy()
        ) {
            let nt = format!(
                "<invalid iri with spaces> <{}> \"{}\" .",
                predicate_part, object_part
            );

            let parser = NTriplesParser::new();
            let nt_clone = nt.clone();
            let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt_clone)).collect();

            // Should fail due to invalid IRI
            prop_assert!(result.is_err(), "Expected error for IRI with spaces");
        }

        #[test]
        fn test_unclosed_literal_fails(
            subject_part in valid_iri_strategy(),
            predicate_part in valid_iri_strategy(),
            object_part in valid_literal_strategy()
        ) {
            let nt = format!(
                "<{}> <{}> \"{} .",
                subject_part, predicate_part, object_part
            );

            let parser = NTriplesParser::new();
            let nt_clone = nt.clone();
            let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt_clone)).collect();

            // Should fail due to unclosed literal
            prop_assert!(result.is_err(), "Expected error for unclosed literal");
        }
    }
}

#[cfg(test)]
mod ntriples_invariants {
    use super::*;

    proptest! {
        /// Invariant: Parsing should never panic
        #[test]
        fn test_parser_never_panics(input in "\\PC*") {
            let parser = NTriplesParser::new();
            let input_owned = input.to_string();
            let _result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(input_owned)).collect();
            // Just checking it doesn't panic
        }

        /// Invariant: Empty input should always succeed with zero triples
        #[test]
        fn test_empty_input_always_succeeds(_dummy in 0..100u32) {
            let parser = NTriplesParser::new();
            let empty = String::new();
            let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(empty)).collect();

            prop_assert!(result.is_ok());
            prop_assert_eq!(result.unwrap().len(), 0);
        }

        /// Invariant: Comment-only lines should not produce triples
        #[test]
        fn test_comments_produce_no_triples(comment_text in "# \\PC*\n") {
            let parser = NTriplesParser::new();
            let comment_owned = comment_text.to_string();
            let result: Result<Vec<_>, _> = parser.for_reader(Cursor::new(comment_owned)).collect();

            if let Ok(triples) = result {
                prop_assert_eq!(triples.len(), 0, "Comments should not produce triples");
            }
        }
    }
}
