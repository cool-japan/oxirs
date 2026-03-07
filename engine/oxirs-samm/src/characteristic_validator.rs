//! SAMM characteristic validation.
//!
//! Validates values against SAMM characteristic definitions including type checks,
//! enumeration membership, collection constraints, measurement unit matching,
//! and sorted-set ordering/uniqueness requirements.

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Canonical SAMM characteristic types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CharacteristicType {
    /// A single, scalar entity value.
    SingleEntity,
    /// An ordered collection (list) of entities.
    Collection,
    /// An unordered set of unique entities.
    Set,
    /// A set sorted by a defined criterion.
    SortedSet,
    /// A time-ordered sequence of measurements.
    TimeSeries,
    /// A measured quantity with a unit of measure.
    Measurement,
    /// A quantifiable value with an optional unit.
    Quantifiable,
    /// A coded value from a defined code system.
    Code,
    /// A duration / time-span value.
    Duration,
    /// A value from a closed enumeration.
    Enumeration,
}

/// Definition of a SAMM characteristic.
#[derive(Debug, Clone)]
pub struct CharacteristicDef {
    /// Human-readable name.
    pub name: String,
    /// Characteristic type.
    pub characteristic_type: CharacteristicType,
    /// XSD/SAMM data-type (e.g. "xsd:string", "xsd:int", "xsd:float").
    pub data_type: String,
    /// Optional unit of measure (required for Measurement/Quantifiable).
    pub unit: Option<String>,
    /// Allowed enumeration values (for Enumeration characteristics).
    pub allowed_values: Vec<String>,
}

impl CharacteristicDef {
    /// Build a minimal definition with defaults.
    pub fn new(
        name: impl Into<String>,
        characteristic_type: CharacteristicType,
        data_type: impl Into<String>,
    ) -> Self {
        CharacteristicDef {
            name: name.into(),
            characteristic_type,
            data_type: data_type.into(),
            unit: None,
            allowed_values: Vec::new(),
        }
    }
}

/// Result of a characteristic validation.
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    /// Whether the value(s) passed validation.
    pub valid: bool,
    /// List of validation error messages.
    pub errors: Vec<String>,
}

impl ValidationResult {
    fn ok() -> Self {
        ValidationResult {
            valid: true,
            errors: Vec::new(),
        }
    }

    fn fail(errors: Vec<String>) -> Self {
        ValidationResult {
            valid: false,
            errors,
        }
    }
}

// ---------------------------------------------------------------------------
// Validator
// ---------------------------------------------------------------------------

/// Stateless SAMM characteristic validator.
pub struct CharacteristicValidator;

impl CharacteristicValidator {
    /// Validate a single value against a characteristic definition.
    pub fn validate_value(value: &str, def: &CharacteristicDef) -> ValidationResult {
        let mut errors = Vec::new();

        // Type-level checks
        if CharacteristicValidator::is_numeric_type(&def.data_type)
            && value.trim().parse::<f64>().is_err()
        {
            errors.push(format!(
                "Value '{}' is not a valid number for data type '{}'",
                value, def.data_type
            ));
        }

        // Enumeration check
        if def.characteristic_type == CharacteristicType::Enumeration
            && !def.allowed_values.is_empty()
            && !def.allowed_values.iter().any(|v| v == value)
        {
            errors.push(format!(
                "Value '{}' is not in the allowed values {:?}",
                value, def.allowed_values
            ));
        }

        if errors.is_empty() {
            ValidationResult::ok()
        } else {
            ValidationResult::fail(errors)
        }
    }

    /// Validate a collection (order-independent, duplicates allowed).
    pub fn validate_collection(values: &[String], def: &CharacteristicDef) -> ValidationResult {
        let mut errors = Vec::new();
        for v in values {
            let r = CharacteristicValidator::validate_value(v.as_str(), def);
            if !r.valid {
                errors.extend(r.errors);
            }
        }
        if errors.is_empty() {
            ValidationResult::ok()
        } else {
            ValidationResult::fail(errors)
        }
    }

    /// Validate a sorted set: values must be ordered (non-decreasing lexicographically)
    /// and unique.
    pub fn validate_sorted_set(values: &[String], def: &CharacteristicDef) -> ValidationResult {
        let mut errors = Vec::new();

        // Check individual values first
        let coll = CharacteristicValidator::validate_collection(values, def);
        if !coll.valid {
            errors.extend(coll.errors);
        }

        // Check uniqueness
        let mut seen = std::collections::HashSet::new();
        for v in values {
            if !seen.insert(v.clone()) {
                errors.push(format!("Duplicate value '{v}' in sorted set"));
            }
        }

        // Check ordering: for numeric types compare numerically; otherwise lexicographic
        if CharacteristicValidator::is_numeric_type(&def.data_type) {
            let nums: Vec<Option<f64>> = values.iter().map(|v| v.parse::<f64>().ok()).collect();
            for i in 1..nums.len() {
                match (nums[i - 1], nums[i]) {
                    (Some(a), Some(b)) if a > b => {
                        errors.push(format!(
                            "Values not in non-decreasing order at position {}: {} > {}",
                            i,
                            values[i - 1],
                            values[i]
                        ));
                    }
                    _ => {}
                }
            }
        } else {
            for i in 1..values.len() {
                if values[i - 1] > values[i] {
                    errors.push(format!(
                        "Values not in non-decreasing order at position {}: '{}' > '{}'",
                        i,
                        values[i - 1],
                        values[i]
                    ));
                }
            }
        }

        if errors.is_empty() {
            ValidationResult::ok()
        } else {
            ValidationResult::fail(errors)
        }
    }

    /// Validate a measurement value with explicit unit.
    pub fn validate_measurement(
        value: &str,
        unit: &str,
        def: &CharacteristicDef,
    ) -> ValidationResult {
        let mut errors = Vec::new();

        // Value must be numeric
        if value.trim().parse::<f64>().is_err() {
            errors.push(format!("Measurement value '{}' is not numeric", value));
        }

        // Unit must match the definition's unit if specified
        if let Some(expected_unit) = &def.unit {
            if unit != expected_unit.as_str() {
                errors.push(format!(
                    "Unit '{}' does not match expected unit '{}'",
                    unit, expected_unit
                ));
            }
        }

        if errors.is_empty() {
            ValidationResult::ok()
        } else {
            ValidationResult::fail(errors)
        }
    }

    /// Validate a value against an enumeration characteristic.
    pub fn validate_enumeration(value: &str, def: &CharacteristicDef) -> ValidationResult {
        if def.allowed_values.is_empty() {
            // No restriction defined
            return ValidationResult::ok();
        }
        if def.allowed_values.iter().any(|v| v == value) {
            ValidationResult::ok()
        } else {
            ValidationResult::fail(vec![format!(
                "Value '{}' is not in allowed enumeration values {:?}",
                value, def.allowed_values
            )])
        }
    }

    /// Return whether two data types are considered compatible.
    pub fn compatible_data_types(type_a: &str, type_b: &str) -> bool {
        if type_a == type_b {
            return true;
        }
        // Both numeric → compatible
        if CharacteristicValidator::is_numeric_type(type_a)
            && CharacteristicValidator::is_numeric_type(type_b)
        {
            return true;
        }
        // Both string → compatible
        if CharacteristicValidator::is_string_type(type_a)
            && CharacteristicValidator::is_string_type(type_b)
        {
            return true;
        }
        false
    }

    /// Return whether `data_type` is a numeric XSD type.
    pub fn is_numeric_type(data_type: &str) -> bool {
        let lower = data_type.to_lowercase();
        lower.contains("int")
            || lower.contains("float")
            || lower.contains("double")
            || lower.contains("decimal")
            || lower.contains("long")
            || lower.contains("short")
            || lower.contains("byte")
            || lower.contains("unsignedint")
            || lower.contains("positiveinteger")
            || lower.contains("nonnegativeinteger")
        // plain "integer" is covered by "int" substring
    }

    /// Return whether `data_type` is a string/text XSD type.
    pub fn is_string_type(data_type: &str) -> bool {
        let lower = data_type.to_lowercase();
        lower.contains("string")
            || lower.contains("normalizedstring")
            || lower.contains("token")
            || lower.contains("language")
            || lower.contains("name")
            || lower.contains("ncname")
            || lower.contains("nmtoken")
            || lower.contains("anyuri")
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn string_def() -> CharacteristicDef {
        CharacteristicDef::new("StringChar", CharacteristicType::SingleEntity, "xsd:string")
    }

    fn int_def() -> CharacteristicDef {
        CharacteristicDef::new("IntChar", CharacteristicType::SingleEntity, "xsd:int")
    }

    fn enum_def(values: &[&str]) -> CharacteristicDef {
        let mut def =
            CharacteristicDef::new("EnumChar", CharacteristicType::Enumeration, "xsd:string");
        def.allowed_values = values.iter().map(|s| s.to_string()).collect();
        def
    }

    fn measurement_def(unit: &str) -> CharacteristicDef {
        let mut def =
            CharacteristicDef::new("MeasChar", CharacteristicType::Measurement, "xsd:float");
        def.unit = Some(unit.to_string());
        def
    }

    fn sorted_int_def() -> CharacteristicDef {
        CharacteristicDef::new("SortedSet", CharacteristicType::SortedSet, "xsd:int")
    }

    // --- validate_value ---
    #[test]
    fn test_validate_string_value_valid() {
        let r = CharacteristicValidator::validate_value("hello", &string_def());
        assert!(r.valid);
    }

    #[test]
    fn test_validate_int_value_valid() {
        let r = CharacteristicValidator::validate_value("42", &int_def());
        assert!(r.valid);
    }

    #[test]
    fn test_validate_int_value_invalid() {
        let r = CharacteristicValidator::validate_value("not-a-number", &int_def());
        assert!(!r.valid);
        assert!(!r.errors.is_empty());
    }

    #[test]
    fn test_validate_float_value_valid() {
        let mut def = int_def();
        def.data_type = "xsd:float".to_string();
        let r = CharacteristicValidator::validate_value("3.14", &def);
        assert!(r.valid);
    }

    #[test]
    fn test_validate_enum_value_in_allowed() {
        let def = enum_def(&["A", "B", "C"]);
        let r = CharacteristicValidator::validate_value("B", &def);
        assert!(r.valid);
    }

    #[test]
    fn test_validate_enum_value_not_in_allowed() {
        let def = enum_def(&["A", "B", "C"]);
        let r = CharacteristicValidator::validate_value("D", &def);
        assert!(!r.valid);
    }

    #[test]
    fn test_validate_enum_empty_allowed_values_passes() {
        let mut def = enum_def(&[]);
        def.characteristic_type = CharacteristicType::Enumeration;
        let r = CharacteristicValidator::validate_value("anything", &def);
        assert!(r.valid);
    }

    // --- validate_collection ---
    #[test]
    fn test_validate_collection_all_valid() {
        let r = CharacteristicValidator::validate_collection(
            &["1".to_string(), "2".to_string(), "3".to_string()],
            &int_def(),
        );
        assert!(r.valid);
    }

    #[test]
    fn test_validate_collection_one_invalid() {
        let r = CharacteristicValidator::validate_collection(
            &["1".to_string(), "abc".to_string()],
            &int_def(),
        );
        assert!(!r.valid);
    }

    #[test]
    fn test_validate_collection_empty_passes() {
        let r = CharacteristicValidator::validate_collection(&[], &int_def());
        assert!(r.valid);
    }

    #[test]
    fn test_validate_collection_allows_duplicates() {
        let r = CharacteristicValidator::validate_collection(
            &["5".to_string(), "5".to_string()],
            &int_def(),
        );
        assert!(r.valid);
    }

    // --- validate_sorted_set ---
    #[test]
    fn test_validate_sorted_set_valid() {
        let r = CharacteristicValidator::validate_sorted_set(
            &["1".to_string(), "2".to_string(), "3".to_string()],
            &sorted_int_def(),
        );
        assert!(r.valid);
    }

    #[test]
    fn test_validate_sorted_set_not_ordered() {
        let r = CharacteristicValidator::validate_sorted_set(
            &["3".to_string(), "1".to_string(), "2".to_string()],
            &sorted_int_def(),
        );
        assert!(!r.valid);
    }

    #[test]
    fn test_validate_sorted_set_duplicate_fails() {
        let r = CharacteristicValidator::validate_sorted_set(
            &["1".to_string(), "2".to_string(), "2".to_string()],
            &sorted_int_def(),
        );
        assert!(!r.valid);
    }

    #[test]
    fn test_validate_sorted_set_empty_passes() {
        let r = CharacteristicValidator::validate_sorted_set(&[], &sorted_int_def());
        assert!(r.valid);
    }

    #[test]
    fn test_validate_sorted_set_single_element() {
        let r =
            CharacteristicValidator::validate_sorted_set(&["42".to_string()], &sorted_int_def());
        assert!(r.valid);
    }

    #[test]
    fn test_validate_sorted_set_string_ordering() {
        let string_set_def =
            CharacteristicDef::new("S", CharacteristicType::SortedSet, "xsd:string");
        let r = CharacteristicValidator::validate_sorted_set(
            &[
                "apple".to_string(),
                "banana".to_string(),
                "cherry".to_string(),
            ],
            &string_set_def,
        );
        assert!(r.valid);
    }

    #[test]
    fn test_validate_sorted_set_string_wrong_order() {
        let string_set_def =
            CharacteristicDef::new("S", CharacteristicType::SortedSet, "xsd:string");
        let r = CharacteristicValidator::validate_sorted_set(
            &["cherry".to_string(), "apple".to_string()],
            &string_set_def,
        );
        assert!(!r.valid);
    }

    // --- validate_measurement ---
    #[test]
    fn test_validate_measurement_correct_unit() {
        let def = measurement_def("kg");
        let r = CharacteristicValidator::validate_measurement("75.5", "kg", &def);
        assert!(r.valid);
    }

    #[test]
    fn test_validate_measurement_wrong_unit() {
        let def = measurement_def("kg");
        let r = CharacteristicValidator::validate_measurement("75.5", "lb", &def);
        assert!(!r.valid);
    }

    #[test]
    fn test_validate_measurement_non_numeric_value() {
        let def = measurement_def("kg");
        let r = CharacteristicValidator::validate_measurement("heavy", "kg", &def);
        assert!(!r.valid);
    }

    #[test]
    fn test_validate_measurement_no_unit_required() {
        let def = CharacteristicDef::new("M", CharacteristicType::Measurement, "xsd:float");
        let r = CharacteristicValidator::validate_measurement("42.0", "any_unit", &def);
        assert!(r.valid);
    }

    // --- validate_enumeration ---
    #[test]
    fn test_validate_enumeration_found() {
        let def = enum_def(&["red", "green", "blue"]);
        let r = CharacteristicValidator::validate_enumeration("green", &def);
        assert!(r.valid);
    }

    #[test]
    fn test_validate_enumeration_not_found() {
        let def = enum_def(&["red", "green", "blue"]);
        let r = CharacteristicValidator::validate_enumeration("yellow", &def);
        assert!(!r.valid);
    }

    // --- compatible_data_types ---
    #[test]
    fn test_compatible_identical_types() {
        assert!(CharacteristicValidator::compatible_data_types(
            "xsd:string",
            "xsd:string"
        ));
    }

    #[test]
    fn test_compatible_numeric_types() {
        assert!(CharacteristicValidator::compatible_data_types(
            "xsd:int",
            "xsd:float"
        ));
    }

    #[test]
    fn test_compatible_string_types() {
        assert!(CharacteristicValidator::compatible_data_types(
            "xsd:string",
            "xsd:normalizedString"
        ));
    }

    #[test]
    fn test_incompatible_string_and_numeric() {
        assert!(!CharacteristicValidator::compatible_data_types(
            "xsd:string",
            "xsd:int"
        ));
    }

    // --- is_numeric_type ---
    #[test]
    fn test_is_numeric_xsd_int() {
        assert!(CharacteristicValidator::is_numeric_type("xsd:int"));
    }

    #[test]
    fn test_is_numeric_xsd_integer() {
        assert!(CharacteristicValidator::is_numeric_type("xsd:integer"));
    }

    #[test]
    fn test_is_numeric_xsd_float() {
        assert!(CharacteristicValidator::is_numeric_type("xsd:float"));
    }

    #[test]
    fn test_is_numeric_xsd_double() {
        assert!(CharacteristicValidator::is_numeric_type("xsd:double"));
    }

    #[test]
    fn test_is_numeric_xsd_decimal() {
        assert!(CharacteristicValidator::is_numeric_type("xsd:decimal"));
    }

    #[test]
    fn test_is_numeric_false_for_string() {
        assert!(!CharacteristicValidator::is_numeric_type("xsd:string"));
    }

    // --- is_string_type ---
    #[test]
    fn test_is_string_xsd_string() {
        assert!(CharacteristicValidator::is_string_type("xsd:string"));
    }

    #[test]
    fn test_is_string_xsd_anyuri() {
        assert!(CharacteristicValidator::is_string_type("xsd:anyURI"));
    }

    #[test]
    fn test_is_string_false_for_int() {
        assert!(!CharacteristicValidator::is_string_type("xsd:int"));
    }
}
