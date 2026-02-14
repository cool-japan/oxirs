//! SAMM to JSON Payload Generator
//!
//! Generates sample JSON payload data from SAMM Aspect models.
//! Uses type-aware random data generation for testing and examples with SciRS2.

use crate::error::SammError;
use crate::metamodel::{Aspect, CharacteristicKind, ModelElement};
use std::time::{SystemTime, UNIX_EPOCH};

// SciRS2 imports for constraint-aware random generation
use scirs2_core::random::{rng, Random, Rng};

/// Generate sample JSON payload from SAMM Aspect
pub fn generate_payload(aspect: &Aspect, _examples: bool) -> Result<String, SammError> {
    let mut json = String::new();
    json.push_str("{\n");

    let properties = aspect.properties();
    for (i, prop) in properties.iter().enumerate() {
        let prop_name = to_snake_case(&prop.name());
        let sample_value = generate_sample_value(prop)?;

        let comma = if i < properties.len() - 1 { "," } else { "" };
        json.push_str(&format!("  \"{}\": {}{}\n", prop_name, sample_value, comma));
    }

    json.push_str("}\n");
    Ok(json)
}

/// Generate sample value based on characteristic data type
fn generate_sample_value(prop: &crate::metamodel::Property) -> Result<String, SammError> {
    if let Some(char) = &prop.characteristic {
        // Check characteristic kind for specialized generation
        match char.kind() {
            CharacteristicKind::Enumeration { values } => {
                // Pick random value from enumeration
                if values.is_empty() {
                    return Ok("\"default\"".to_string());
                }
                let idx = get_pseudo_random_index(values.len());
                return Ok(format!(
                    "\"{}\"",
                    values.get(idx).unwrap_or(&"default".to_string())
                ));
            }
            CharacteristicKind::State {
                values,
                default_value,
            } => {
                // Use default value if available, otherwise pick first
                if let Some(default) = default_value {
                    return Ok(format!("\"{}\"", default));
                }
                return Ok(format!(
                    "\"{}\"",
                    values.first().unwrap_or(&"unknown".to_string())
                ));
            }
            CharacteristicKind::Measurement { .. } | CharacteristicKind::Quantifiable { .. } => {
                // Generate numeric value with unit awareness
                if let Some(dt) = &char.data_type {
                    return Ok(generate_numeric_value(dt));
                }
            }
            CharacteristicKind::Collection { .. } | CharacteristicKind::List { .. } => {
                // Generate array of sample values
                if let Some(dt) = &char.data_type {
                    let sample1 = generate_value_for_xsd_type(dt);
                    let sample2 = generate_value_for_xsd_type(dt);
                    return Ok(format!("[{}, {}]", sample1, sample2));
                }
            }
            CharacteristicKind::TimeSeries { .. } => {
                return Ok("[{\"timestamp\": \"2025-10-11T12:00:00Z\", \"value\": 42}]".to_string());
            }
            _ => {}
        }

        // Fallback to data type-based generation
        if let Some(dt) = &char.data_type {
            return Ok(generate_value_for_xsd_type(dt));
        }
    }

    // Default to string
    Ok(format!("\"sample_{}\"", to_snake_case(&prop.name())))
}

/// Generate numeric value with randomization
fn generate_numeric_value(xsd_type: &str) -> String {
    let rand_val = get_pseudo_random_f64();

    match xsd_type {
        t if t.ends_with("int") | t.ends_with("integer") => {
            format!("{}", 1 + (rand_val * 99.0) as i32)
        }
        t if t.ends_with("long") => {
            format!("{}", 1000 + (rand_val * 99000.0) as i64)
        }
        t if t.ends_with("short") | t.ends_with("byte") => {
            format!("{}", 1 + (rand_val * 49.0) as i16)
        }
        t if t.ends_with("decimal") => {
            format!("{:.2}", 1.0 + rand_val * 999.0 / 10.0)
        }
        t if t.ends_with("float") => {
            format!("{:.2}", 1.0 + rand_val * 99.0 / 10.0)
        }
        t if t.ends_with("double") => {
            format!("{:.6}", 1.0 + rand_val * 99.0 / 10.0)
        }
        _ => "42".to_string(),
    }
}

/// Get pseudo-random f64 value between 0.0 and 1.0
fn get_pseudo_random_f64() -> f64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (nanos % 1000) as f64 / 1000.0
}

/// Get pseudo-random index for array
fn get_pseudo_random_index(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    (get_pseudo_random_f64() * len as f64) as usize % len
}

/// Generate sample value for XSD data type
fn generate_value_for_xsd_type(xsd_type: &str) -> String {
    match xsd_type {
        t if t.ends_with("string") => "\"sample_string\"".to_string(),
        t if t.ends_with("int") | t.ends_with("integer") => "42".to_string(),
        t if t.ends_with("long") => "1000000".to_string(),
        t if t.ends_with("short") | t.ends_with("byte") => "10".to_string(),
        t if t.ends_with("decimal") => "123.45".to_string(),
        t if t.ends_with("float") => "3.14".to_string(),
        t if t.ends_with("double") => "3.141592653589793".to_string(),
        t if t.ends_with("boolean") => "true".to_string(),
        t if t.ends_with("date") => "\"2025-10-11\"".to_string(),
        t if t.ends_with("dateTime") | t.ends_with("dateTimeStamp") => {
            "\"2025-10-11T12:00:00Z\"".to_string()
        }
        t if t.ends_with("time") => "\"12:00:00\"".to_string(),
        t if t.ends_with("duration") => "\"PT1H30M\"".to_string(),
        t if t.ends_with("anyURI") => "\"https://example.com\"".to_string(),
        t if t.ends_with("base64Binary") => "\"SGVsbG8gV29ybGQ=\"".to_string(),
        t if t.ends_with("hexBinary") => "\"48656c6c6f\"".to_string(),
        t if t.ends_with("gYear") => "\"2025\"".to_string(),
        t if t.ends_with("gYearMonth") => "\"2025-10\"".to_string(),
        t if t.ends_with("gMonth") => "\"--10\"".to_string(),
        t if t.ends_with("gMonthDay") => "\"--10-11\"".to_string(),
        t if t.ends_with("gDay") => "\"---11\"".to_string(),
        _ => "\"unknown_type\"".to_string(),
    }
}

/// Advanced payload generation with constraints using SciRS2
pub fn generate_value_with_constraints(
    xsd_type: &str,
    min: Option<f64>,
    max: Option<f64>,
    pattern: Option<&str>,
) -> String {
    let mut rng = rng();

    // Handle pattern-based constraints first
    if let Some(pat) = pattern {
        if let Some(value) = generate_from_pattern(pat) {
            return value;
        }
    }

    // Handle numeric types with min/max constraints using scirs2-core random API
    if let (Some(min_val), Some(max_val)) = (min, max) {
        return match xsd_type {
            t if t.ends_with("int") | t.ends_with("integer") => {
                let value = rng.random_range(min_val as i32..max_val as i32);
                format!("{}", value)
            }
            t if t.ends_with("long") => {
                let value = rng.random_range(min_val as i64..max_val as i64);
                format!("{}", value)
            }
            t if t.ends_with("short") | t.ends_with("byte") => {
                let value = rng.random_range(min_val as i16..max_val as i16);
                format!("{}", value)
            }
            t if t.ends_with("decimal") | t.ends_with("float") | t.ends_with("double") => {
                let value = rng.random_range(min_val..max_val);
                format!("{:.2}", value)
            }
            _ => generate_value_for_xsd_type(xsd_type),
        };
    }

    // Handle min-only constraint
    if let Some(min_val) = min {
        return match xsd_type {
            t if t.ends_with("int") | t.ends_with("integer") => {
                let max_range = (min_val as i32).saturating_add(1000);
                let value = rng.random_range(min_val as i32..max_range);
                format!("{}", value)
            }
            t if t.ends_with("long") => {
                let max_range = (min_val as i64).saturating_add(10000);
                let value = rng.random_range(min_val as i64..max_range);
                format!("{}", value)
            }
            t if t.ends_with("decimal") | t.ends_with("float") | t.ends_with("double") => {
                let value = rng.random_range(min_val..(min_val + 1000.0));
                format!("{:.2}", value)
            }
            _ => generate_value_for_xsd_type(xsd_type),
        };
    }

    // Handle max-only constraint
    if let Some(max_val) = max {
        return match xsd_type {
            t if t.ends_with("int") | t.ends_with("integer") => {
                let value = rng.random_range(0..(max_val as i32));
                format!("{}", value)
            }
            t if t.ends_with("long") => {
                let value = rng.random_range(0..(max_val as i64));
                format!("{}", value)
            }
            t if t.ends_with("decimal") | t.ends_with("float") | t.ends_with("double") => {
                let value = rng.random_range(0.0..max_val);
                format!("{:.2}", value)
            }
            _ => generate_value_for_xsd_type(xsd_type),
        };
    }

    // Fallback to basic generation
    generate_value_for_xsd_type(xsd_type)
}

/// Generate value from regex pattern (simplified pattern matching)
fn generate_from_pattern(pattern: &str) -> Option<String> {
    // Handle common SAMM patterns
    match pattern {
        // Email pattern
        p if p.contains("@") || p.contains("email") => Some("\"example@domain.com\"".to_string()),
        // URL pattern
        p if p.contains("http") || p.contains("url") => Some("\"https://example.com\"".to_string()),
        // Phone number pattern
        p if p.contains("\\d{3}-\\d{4}") || p.contains("phone") => Some("\"123-4567\"".to_string()),
        // UUID pattern
        p if p.contains("[0-9a-f]{8}-[0-9a-f]{4}") || p.contains("uuid") => {
            Some("\"550e8400-e29b-41d4-a716-446655440000\"".to_string())
        }
        // ISBN pattern
        p if p.contains("isbn") => Some("\"978-0-13-468599-1\"".to_string()),
        // Date pattern (YYYY-MM-DD)
        p if p.contains("\\d{4}-\\d{2}-\\d{2}") => Some("\"2025-10-31\"".to_string()),
        // Time pattern (HH:MM:SS)
        p if p.contains("\\d{2}:\\d{2}:\\d{2}") => Some("\"12:00:00\"".to_string()),
        // Hex color pattern
        p if p.contains("#[0-9a-fA-F]{6}") => Some("\"#FF5733\"".to_string()),
        // IP address pattern
        p if p.contains("\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}") => {
            Some("\"192.168.1.1\"".to_string())
        }
        _ => None,
    }
}

/// Generate multiple example payloads (future enhancement)
#[allow(dead_code)]
pub fn generate_multiple_payloads(aspect: &Aspect, count: usize) -> Result<Vec<String>, SammError> {
    let mut payloads = Vec::new();

    for _ in 0..count {
        payloads.push(generate_payload(aspect, false)?);
    }

    Ok(payloads)
}

/// Convert PascalCase/camelCase to snake_case
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(
                ch.to_lowercase()
                    .next()
                    .expect("lowercase should produce a character"),
            );
        } else {
            result.push(ch);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xsd_type_value_generation() {
        assert_eq!(
            generate_value_for_xsd_type("http://www.w3.org/2001/XMLSchema#string"),
            "\"sample_string\""
        );
        assert_eq!(
            generate_value_for_xsd_type("http://www.w3.org/2001/XMLSchema#int"),
            "42"
        );
        assert_eq!(
            generate_value_for_xsd_type("http://www.w3.org/2001/XMLSchema#boolean"),
            "true"
        );
    }

    #[test]
    fn test_snake_case_conversion() {
        assert_eq!(to_snake_case("MovementAspect"), "movement_aspect");
        assert_eq!(to_snake_case("currentSpeed"), "current_speed");
    }

    #[test]
    fn test_constraint_aware_generation_with_min_max() {
        let xsd_type = "http://www.w3.org/2001/XMLSchema#int";

        // Test with min and max constraints
        let value = generate_value_with_constraints(xsd_type, Some(10.0), Some(20.0), None);
        let num: i32 = value.parse().expect("Should be valid integer");
        assert!(
            (10..=20).contains(&num),
            "Value {} should be in range [10, 20]",
            num
        );
    }

    #[test]
    fn test_constraint_aware_generation_with_min_only() {
        let xsd_type = "http://www.w3.org/2001/XMLSchema#long";

        // Test with min constraint only
        let value = generate_value_with_constraints(xsd_type, Some(1000.0), None, None);
        let num: i64 = value.parse().expect("Should be valid long");
        assert!(num >= 1000, "Value {} should be >= 1000", num);
    }

    #[test]
    fn test_constraint_aware_generation_with_max_only() {
        let xsd_type = "http://www.w3.org/2001/XMLSchema#int";

        // Test with max constraint only
        let value = generate_value_with_constraints(xsd_type, None, Some(100.0), None);
        let num: i32 = value.parse().expect("Should be valid integer");
        assert!(num <= 100, "Value {} should be <= 100", num);
    }

    #[test]
    fn test_constraint_aware_generation_with_decimal() {
        let xsd_type = "http://www.w3.org/2001/XMLSchema#decimal";

        // Test decimal with range
        let value = generate_value_with_constraints(xsd_type, Some(0.0), Some(1.0), None);
        let num: f64 = value.parse().expect("Should be valid decimal");
        assert!(
            (0.0..=1.0).contains(&num),
            "Value {} should be in range [0.0, 1.0]",
            num
        );
    }

    #[test]
    fn test_pattern_generation_email() {
        let pattern = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$";
        let value = generate_from_pattern(pattern);
        assert!(value.is_some());
        assert!(value.expect("operation should succeed").contains("@"));
    }

    #[test]
    fn test_pattern_generation_url() {
        let pattern = "^https?://.*";
        let value = generate_from_pattern(pattern);
        assert!(value.is_some());
        assert!(value
            .expect("operation should succeed")
            .contains("https://"));
    }

    #[test]
    fn test_pattern_generation_phone() {
        let pattern = "\\d{3}-\\d{4}";
        let value = generate_from_pattern(pattern);
        assert!(value.is_some());
        assert!(value.expect("operation should succeed").contains("-"));
    }

    #[test]
    fn test_pattern_generation_uuid() {
        let pattern = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";
        let value = generate_from_pattern(pattern);
        assert!(value.is_some());
        // UUID format: 8-4-4-4-12 hex digits
        let uuid = value.expect("operation should succeed");
        assert!(uuid.contains("550e8400"));
    }

    #[test]
    fn test_pattern_generation_date() {
        let pattern = "\\d{4}-\\d{2}-\\d{2}";
        let value = generate_from_pattern(pattern);
        assert!(value.is_some());
        assert!(value.expect("operation should succeed").contains("2025"));
    }

    #[test]
    fn test_pattern_generation_ip_address() {
        let pattern = "\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}";
        let value = generate_from_pattern(pattern);
        assert!(value.is_some());
        assert!(value.expect("operation should succeed").contains("192.168"));
    }

    #[test]
    fn test_pattern_generation_hex_color() {
        let pattern = "#[0-9a-fA-F]{6}";
        let value = generate_from_pattern(pattern);
        assert!(value.is_some());
        let color = value.expect("operation should succeed");
        assert!(color.starts_with("\"#"));
    }

    #[test]
    fn test_constraint_with_pattern_priority() {
        let xsd_type = "http://www.w3.org/2001/XMLSchema#string";

        // Pattern should take priority over min/max for strings
        let value =
            generate_value_with_constraints(xsd_type, Some(10.0), Some(20.0), Some("email"));
        assert!(value.contains("@"));
    }
}
