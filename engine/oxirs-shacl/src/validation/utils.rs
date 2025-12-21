//! Utility functions for validation operations

use crate::{Result, ShaclError};
use oxirs_core::model::Term;

/// Format a term for use in SPARQL queries
pub fn format_term_for_sparql(term: &Term) -> Result<String> {
    match term {
        Term::NamedNode(node) => Ok(format!("<{}>", node.as_str())),
        Term::BlankNode(node) => Ok(node.as_str().to_string()),
        Term::Literal(literal) => {
            // Properly format literal with datatype and language tags
            let mut result = format!(
                "\"{}\"",
                literal.value().replace('\\', "\\\\").replace('"', "\\\"")
            );

            if let Some(lang) = literal.language() {
                result.push('@');
                result.push_str(lang);
            } else {
                let datatype = literal.datatype();
                let dt_str = datatype.as_str();
                // Only add datatype if it's not the default xsd:string
                if dt_str != "http://www.w3.org/2001/XMLSchema#string" {
                    result.push_str("^^<");
                    result.push_str(dt_str);
                    result.push('>');
                }
            }

            Ok(result)
        }
        Term::Variable(var) => Ok(format!("?{}", var.name())),
        Term::QuotedTriple(_) => Err(ShaclError::ValidationEngine(
            "Quoted triples not supported in validation queries".to_string(),
        )),
    }
}

/// Format a term for display in error messages
pub fn format_term_for_message(term: &Term) -> String {
    match term {
        Term::NamedNode(node) => node.as_str().to_string(),
        Term::BlankNode(node) => node.as_str().to_string(),
        Term::Literal(literal) => {
            if let Some(lang) = literal.language() {
                format!("\"{}\"@{}", literal.value(), lang)
            } else {
                format!("\"{}\"", literal.value())
            }
        }
        Term::Variable(var) => format!("?{}", var.name()),
        Term::QuotedTriple(_) => "<<quoted_triple>>".to_string(),
    }
}

/// Convert a term to a string representation suitable for sorting and comparison
pub fn term_to_sort_key(term: &Term) -> String {
    match term {
        Term::NamedNode(node) => format!("iri:{}", node.as_str()),
        Term::BlankNode(node) => format!("blank:{}", node.as_str()),
        Term::Literal(literal) => {
            let base = format!("literal:{}", literal.value());
            if let Some(lang) = literal.language() {
                format!("{base}@{lang}")
            } else {
                let datatype = literal.datatype();
                format!("{}^^{}", base, datatype.as_str())
            }
        }
        Term::Variable(var) => format!("var:{}", var.name()),
        Term::QuotedTriple(_) => "quoted:<<>>".to_string(),
    }
}

/// Check if a term represents a numeric value
pub fn is_numeric_term(term: &Term) -> bool {
    if let Term::Literal(literal) = term {
        let datatype = literal.datatype();
        let dt_str = datatype.as_str();
        matches!(
            dt_str,
            "http://www.w3.org/2001/XMLSchema#integer"
                | "http://www.w3.org/2001/XMLSchema#decimal"
                | "http://www.w3.org/2001/XMLSchema#float"
                | "http://www.w3.org/2001/XMLSchema#double"
                | "http://www.w3.org/2001/XMLSchema#byte"
                | "http://www.w3.org/2001/XMLSchema#short"
                | "http://www.w3.org/2001/XMLSchema#int"
                | "http://www.w3.org/2001/XMLSchema#long"
                | "http://www.w3.org/2001/XMLSchema#unsignedByte"
                | "http://www.w3.org/2001/XMLSchema#unsignedShort"
                | "http://www.w3.org/2001/XMLSchema#unsignedInt"
                | "http://www.w3.org/2001/XMLSchema#unsignedLong"
                | "http://www.w3.org/2001/XMLSchema#positiveInteger"
                | "http://www.w3.org/2001/XMLSchema#nonPositiveInteger"
                | "http://www.w3.org/2001/XMLSchema#negativeInteger"
                | "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
        ) || literal.value().parse::<f64>().is_ok() // Also try to parse as number for untyped literals
    } else {
        false
    }
}

/// Check if a term represents a boolean value
pub fn is_boolean_term(term: &Term) -> bool {
    if let Term::Literal(literal) = term {
        let datatype = literal.datatype();
        datatype.as_str() == "http://www.w3.org/2001/XMLSchema#boolean"
            || matches!(literal.value(), "true" | "false")
    } else {
        false
    }
}

/// Check if a term represents a date/time value
pub fn is_datetime_term(term: &Term) -> bool {
    if let Term::Literal(literal) = term {
        let datatype = literal.datatype();
        let dt_str = datatype.as_str();
        matches!(
            dt_str,
            "http://www.w3.org/2001/XMLSchema#dateTime"
                | "http://www.w3.org/2001/XMLSchema#date"
                | "http://www.w3.org/2001/XMLSchema#time"
                | "http://www.w3.org/2001/XMLSchema#gYear"
                | "http://www.w3.org/2001/XMLSchema#gYearMonth"
                | "http://www.w3.org/2001/XMLSchema#gMonth"
                | "http://www.w3.org/2001/XMLSchema#gMonthDay"
                | "http://www.w3.org/2001/XMLSchema#gDay"
        )
    } else {
        false
    }
}

/// Parse a numeric value from a term
pub fn parse_numeric_value(term: &Term) -> Result<f64> {
    if let Term::Literal(literal) = term {
        literal.value().parse::<f64>().map_err(|e| {
            ShaclError::ValidationEngine(format!(
                "Failed to parse numeric value '{}': {}",
                literal.value(),
                e
            ))
        })
    } else {
        Err(ShaclError::ValidationEngine(format!(
            "Term is not a literal: {}",
            format_term_for_message(term)
        )))
    }
}

/// Parse a boolean value from a term
pub fn parse_boolean_value(term: &Term) -> Result<bool> {
    if let Term::Literal(literal) = term {
        match literal.value() {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            _ => Err(ShaclError::ValidationEngine(format!(
                "Invalid boolean value: '{}'",
                literal.value()
            ))),
        }
    } else {
        Err(ShaclError::ValidationEngine(format!(
            "Term is not a literal: {}",
            format_term_for_message(term)
        )))
    }
}

/// Normalize a string for comparison (trim, lowercase, etc.)
pub fn normalize_string_for_comparison(s: &str) -> String {
    s.trim().to_lowercase()
}

/// Check if two terms are equivalent (considering type coercion)
pub fn terms_equivalent(term1: &Term, term2: &Term) -> bool {
    // Direct equality first
    if term1 == term2 {
        return true;
    }

    // Try numeric comparison if both are numeric
    if is_numeric_term(term1) && is_numeric_term(term2) {
        if let (Ok(val1), Ok(val2)) = (parse_numeric_value(term1), parse_numeric_value(term2)) {
            return (val1 - val2).abs() < f64::EPSILON;
        }
    }

    // Try boolean comparison if both are boolean
    if is_boolean_term(term1) && is_boolean_term(term2) {
        if let (Ok(val1), Ok(val2)) = (parse_boolean_value(term1), parse_boolean_value(term2)) {
            return val1 == val2;
        }
    }

    false
}

/// Escape a string for use in regular expressions
pub fn escape_regex(input: &str) -> String {
    regex::escape(input)
}

/// Convert a term to its canonical string representation
pub fn term_to_canonical_string(term: &Term) -> String {
    match term {
        Term::NamedNode(node) => format!("<{}>", node.as_str()),
        Term::BlankNode(node) => format!("_:{}", node.as_str()),
        Term::Literal(literal) => {
            let mut result = format!(
                "\"{}\"",
                literal.value().replace('\\', "\\\\").replace('"', "\\\"")
            );
            if let Some(lang) = literal.language() {
                result.push('@');
                result.push_str(lang);
            } else {
                let datatype = literal.datatype();
                result.push_str("^^<");
                result.push_str(datatype.as_str());
                result.push('>');
            }
            result
        }
        Term::Variable(var) => format!("?{}", var.name()),
        Term::QuotedTriple(_) => "<<quoted_triple>>".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{Literal, NamedNode};

    #[test]
    fn test_format_term_for_message() {
        let iri_term = Term::NamedNode(NamedNode::new("http://example.org/test").unwrap());
        assert_eq!(
            format_term_for_message(&iri_term),
            "http://example.org/test"
        );

        let literal_term = Term::Literal(Literal::new("hello"));
        assert_eq!(format_term_for_message(&literal_term), "\"hello\"");

        let lang_literal = Term::Literal(Literal::new_lang("hello", "en").unwrap());
        assert_eq!(format_term_for_message(&lang_literal), "\"hello\"@en");
    }

    #[test]
    fn test_is_numeric_term() {
        let int_literal = Term::Literal(Literal::new_typed(
            "42",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
        ));
        assert!(is_numeric_term(&int_literal));

        let string_literal = Term::Literal(Literal::new("hello"));
        assert!(!is_numeric_term(&string_literal));

        let iri_term = Term::NamedNode(NamedNode::new("http://example.org/test").unwrap());
        assert!(!is_numeric_term(&iri_term));
    }

    #[test]
    fn test_parse_numeric_value() {
        let int_literal = Term::Literal(Literal::new_typed(
            "42",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
        ));
        assert_eq!(parse_numeric_value(&int_literal).unwrap(), 42.0);

        let float_literal = Term::Literal(Literal::new_typed(
            "3.15",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#float").unwrap(),
        ));
        assert!((parse_numeric_value(&float_literal).unwrap() - 3.15).abs() < f64::EPSILON);
    }

    #[test]
    fn test_terms_equivalent() {
        let term1 = Term::Literal(Literal::new_typed(
            "42",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
        ));
        let term2 = Term::Literal(Literal::new_typed(
            "42.0",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal").unwrap(),
        ));

        assert!(terms_equivalent(&term1, &term2));
    }
}
