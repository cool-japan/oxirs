//! SAMM to JSON Payload Generator
//!
//! Generates sample JSON payload data from SAMM Aspect models.
//! Uses type-aware random data generation for testing and examples.

use crate::error::SammError;
use crate::metamodel::{Aspect, CharacteristicKind, ModelElement};
use std::time::{SystemTime, UNIX_EPOCH};

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

/// Advanced payload generation with constraints (future enhancement)
#[allow(dead_code)]
fn generate_value_with_constraints(
    xsd_type: &str,
    _min: Option<f64>,
    _max: Option<f64>,
    _pattern: Option<&str>,
) -> String {
    // TODO: Implement constraint-aware generation
    // For alpha.3, use basic generation
    generate_value_for_xsd_type(xsd_type)
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
            result.push(ch.to_lowercase().next().unwrap());
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
}
