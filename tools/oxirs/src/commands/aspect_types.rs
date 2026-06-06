//! Types and utility functions for SAMM Aspect Model processing
// These utilities are used by aspect_tests and are kept for compatibility.
#![allow(dead_code)]

/// Map XSD data types to Rust types
pub(crate) fn map_xsd_to_rust(xsd_type: &str) -> String {
    match xsd_type {
        t if t.ends_with("string") => "String".to_string(),
        t if t.ends_with("int") || t.ends_with("integer") => "i32".to_string(),
        t if t.ends_with("long") => "i64".to_string(),
        t if t.ends_with("float") => "f32".to_string(),
        t if t.ends_with("double") => "f64".to_string(),
        t if t.ends_with("boolean") => "bool".to_string(),
        t if t.ends_with("date") => "chrono::NaiveDate".to_string(),
        t if t.ends_with("dateTime") => "chrono::DateTime<chrono::Utc>".to_string(),
        _ => "String".to_string(),
    }
}

/// Map XSD data types to JSON Schema types with format attributes
pub(crate) fn map_xsd_to_json_schema(xsd_type: &str) -> (String, Option<String>) {
    match xsd_type {
        t if t.ends_with("string") => ("string".to_string(), None),
        t if t.ends_with("int") || t.ends_with("integer") => {
            ("integer".to_string(), Some("int32".to_string()))
        }
        t if t.ends_with("long") => ("integer".to_string(), Some("int64".to_string())),
        t if t.ends_with("float") => ("number".to_string(), Some("float".to_string())),
        t if t.ends_with("double") => ("number".to_string(), Some("double".to_string())),
        t if t.ends_with("boolean") => ("boolean".to_string(), None),
        t if t.ends_with("date") => ("string".to_string(), Some("date".to_string())),
        t if t.ends_with("dateTime") => ("string".to_string(), Some("date-time".to_string())),
        t if t.ends_with("time") => ("string".to_string(), Some("time".to_string())),
        t if t.ends_with("duration") => ("string".to_string(), Some("duration".to_string())),
        t if t.ends_with("anyURI") => ("string".to_string(), Some("uri".to_string())),
        t if t.ends_with("byte") => ("integer".to_string(), None),
        t if t.ends_with("short") => ("integer".to_string(), Some("int32".to_string())),
        t if t.ends_with("decimal") => ("number".to_string(), None),
        _ => ("string".to_string(), None),
    }
}

/// Convert PascalCase/camelCase to snake_case
pub(crate) fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(
                ch.to_lowercase()
                    .next()
                    .expect("lowercase conversion should produce at least one char"),
            );
        } else {
            result.push(ch);
        }
    }
    result
}
