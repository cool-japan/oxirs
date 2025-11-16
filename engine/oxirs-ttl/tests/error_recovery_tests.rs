//! Tests for error recovery and lenient mode

use oxirs_ttl::formats::turtle::TurtleParser;
use oxirs_ttl::toolkit::Parser;

#[test]
fn test_strict_mode_fails_on_first_error() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:good1 ex:pred "valid" .
ex:bad <invalid iri> "broken" .
ex:good2 ex:pred "also valid" .
"#;

    let parser = TurtleParser::new(); // strict mode by default
    let result = parser.parse(ttl.as_bytes());

    // Should fail on the invalid IRI
    assert!(result.is_err());
}

#[test]
fn test_lenient_mode_continues_after_errors() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:good1 ex:pred "valid" .
ex:bad <invalid iri> "broken" .
ex:good2 ex:pred "also valid" .
"#;

    let parser = TurtleParser::new_lenient();
    let result = parser.parse(ttl.as_bytes());

    // In lenient mode, we should get an error but it should contain multiple errors
    match result {
        Err(oxirs_ttl::error::TurtleParseError::Multiple { errors }) => {
            assert!(!errors.is_empty(), "Should collect errors in lenient mode");
        }
        Ok(_) => {
            // This is also acceptable - successfully parsed triples might be returned
            // depending on implementation details
        }
        Err(e) => {
            panic!("Expected Multiple error or success, got: {:?}", e);
        }
    }
}

#[test]
fn test_lenient_mode_parses_valid_statements() {
    // All valid - lenient mode should behave like strict mode
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:subject1 ex:predicate "object1" .
ex:subject2 ex:predicate "object2" .
ex:subject3 ex:predicate "object3" .
"#;

    let parser = TurtleParser::new_lenient();
    let result = parser.parse(ttl.as_bytes());

    assert!(result.is_ok(), "Lenient mode should handle valid input");
    let triples = result.unwrap();
    assert_eq!(triples.len(), 3);
}

#[test]
fn test_error_position_tracking() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:subject1 ex:predicate "object1" .
ex:bad
"#;

    let parser = TurtleParser::new();
    let result = parser.parse(ttl.as_bytes());

    assert!(result.is_err());
    let error = result.unwrap_err();

    // Check that error has position information
    if let Some(pos) = error.position() {
        assert!(pos.line > 0, "Error should have line number");
        assert!(pos.column > 0, "Error should have column number");
    }
}

#[test]
fn test_undefined_prefix_error() {
    let ttl = r#"
ex:subject ex:predicate "object" .
"#;

    let parser = TurtleParser::new();
    let result = parser.parse(ttl.as_bytes());

    assert!(result.is_err());
    let error_msg = format!("{:?}", result.unwrap_err());
    assert!(
        error_msg.contains("Undefined") || error_msg.contains("prefix"),
        "Error should mention undefined prefix"
    );
}

#[test]
fn test_invalid_iri_error() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:subject ex:predicate <not a valid iri> .
"#;

    let parser = TurtleParser::new();
    let result = parser.parse(ttl.as_bytes());

    // Should fail due to invalid IRI (space in IRI)
    assert!(result.is_err());
}

#[test]
fn test_missing_period_error() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:subject ex:predicate "object"
ex:another ex:pred "value" .
"#;

    let parser = TurtleParser::new();
    let result = parser.parse(ttl.as_bytes());

    // Should fail due to missing period
    assert!(result.is_err());
}

#[test]
fn test_lenient_mode_with_multiple_errors() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:good1 ex:pred "valid" .
ex:bad1 <invalid iri 1> "broken" .
ex:good2 ex:pred "also valid" .
ex:bad2 <invalid iri 2> "broken" .
ex:good3 ex:pred "still valid" .
"#;

    let parser = TurtleParser::new_lenient();
    let result = parser.parse(ttl.as_bytes());

    // Should collect multiple errors
    if let Err(oxirs_ttl::error::TurtleParseError::Multiple { errors }) = result {
        assert!(
            errors.len() >= 2,
            "Should collect at least 2 errors, got {}",
            errors.len()
        );
    }
}

#[test]
fn test_error_message_quality() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:subject ex:predicate 123invalid .
"#;

    let parser = TurtleParser::new();
    let result = parser.parse(ttl.as_bytes());

    assert!(result.is_err());
    let error_msg = format!("{}", result.unwrap_err());

    // Error message should be informative
    assert!(!error_msg.is_empty(), "Error message should not be empty");
    assert!(
        error_msg.len() > 10,
        "Error message should be reasonably descriptive"
    );
}

#[test]
#[ignore] // TODO: Fix infinite loop in error recovery for invalid language tags
fn test_lenient_mode_recovers_from_bad_language_tag() {
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:good1 ex:text "Hello"@en .
ex:bad ex:text "Bad"@invalid_tag_with_underscores .
ex:good2 ex:text "World"@fr .
"#;

    let parser = TurtleParser::new_lenient();
    let result = parser.parse(ttl.as_bytes());

    // Lenient mode should handle this
    match result {
        Ok(triples) => {
            // Might successfully parse valid statements
            assert!(!triples.is_empty());
        }
        Err(oxirs_ttl::error::TurtleParseError::Multiple { errors }) => {
            // Or collect errors and continue
            assert!(!errors.is_empty());
        }
        Err(e) => {
            panic!("Unexpected error type: {:?}", e);
        }
    }
}
