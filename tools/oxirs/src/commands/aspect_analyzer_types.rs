//! Type mappings, format enums, XSD-to-Rust helpers, conversion helpers for aspect_analyzer

/// Map XSD datatype URI to Rust type name
pub fn map_xsd_to_rust(xsd_type: &str) -> String {
    match xsd_type {
        t if t.ends_with("string") => "String".to_string(),
        t if t.ends_with("integer") => "i64".to_string(),
        t if t.ends_with("int") => "i32".to_string(),
        t if t.ends_with("long") => "i64".to_string(),
        t if t.ends_with("short") => "i16".to_string(),
        t if t.ends_with("byte") => "i8".to_string(),
        t if t.ends_with("unsignedInt") => "u32".to_string(),
        t if t.ends_with("unsignedLong") => "u64".to_string(),
        t if t.ends_with("unsignedShort") => "u16".to_string(),
        t if t.ends_with("unsignedByte") => "u8".to_string(),
        t if t.ends_with("decimal") => "f64".to_string(),
        t if t.ends_with("float") => "f32".to_string(),
        t if t.ends_with("double") => "f64".to_string(),
        t if t.ends_with("boolean") => "bool".to_string(),
        t if t.ends_with("date") => "chrono::NaiveDate".to_string(),
        t if t.ends_with("dateTime") => "chrono::DateTime<chrono::Utc>".to_string(),
        t if t.ends_with("time") => "chrono::NaiveTime".to_string(),
        t if t.ends_with("duration") => "std::time::Duration".to_string(),
        t if t.ends_with("base64Binary") => "Vec<u8>".to_string(),
        t if t.ends_with("hexBinary") => "Vec<u8>".to_string(),
        t if t.ends_with("anyURI") => "url::Url".to_string(),
        _ => "String".to_string(),
    }
}

/// Map XSD datatype URI to JSON Schema type and optional format attribute
pub fn map_xsd_to_json_schema(xsd_type: &str) -> (String, Option<String>) {
    match xsd_type {
        t if t.ends_with("string") => ("string".to_string(), None),
        t if t.ends_with("integer")
            || t.ends_with("int")
            || t.ends_with("long")
            || t.ends_with("short")
            || t.ends_with("byte")
            || t.ends_with("unsignedInt")
            || t.ends_with("unsignedLong")
            || t.ends_with("unsignedShort")
            || t.ends_with("unsignedByte") =>
        {
            ("integer".to_string(), None)
        }
        t if t.ends_with("decimal") || t.ends_with("float") || t.ends_with("double") => {
            ("number".to_string(), None)
        }
        t if t.ends_with("boolean") => ("boolean".to_string(), None),
        t if t.ends_with("dateTime") => ("string".to_string(), Some("date-time".to_string())),
        t if t.ends_with("date") => ("string".to_string(), Some("date".to_string())),
        t if t.ends_with("time") => ("string".to_string(), Some("time".to_string())),
        t if t.ends_with("anyURI") => ("string".to_string(), Some("uri".to_string())),
        t if t.ends_with("base64Binary") => ("string".to_string(), Some("byte".to_string())),
        _ => ("string".to_string(), None),
    }
}

/// Convert a PascalCase/camelCase string to snake_case
pub fn to_snake_case(name: &str) -> String {
    let mut result = String::new();
    let mut chars = name.chars().peekable();

    while let Some(c) = chars.next() {
        if c.is_uppercase() {
            if !result.is_empty() {
                // Check if next char is lowercase (camelCase boundary)
                if chars.peek().map(|nc| nc.is_lowercase()).unwrap_or(false) {
                    result.push('_');
                }
            }
            result.push(c.to_lowercase().next().unwrap_or(c));
        } else {
            result.push(c);
        }
    }

    result
}
