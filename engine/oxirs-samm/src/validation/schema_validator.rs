//! SAMM Schema Validator
//!
//! Validates SAMM Aspect Model definitions against the SAMM specification.
//! Enforces structural constraints, naming conventions, and semantic rules
//! from the SAMM 2.3.0 specification.

use crate::error::{Result, SammError};
use crate::metamodel::{
    Aspect, Characteristic, CharacteristicKind, Entity, ModelElement, Property,
};

/// A report produced after validating a SAMM Aspect model.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Whether the model passed all mandatory validation rules.
    pub is_valid: bool,
    /// Errors that cause the model to be invalid.
    pub errors: Vec<SchemaValidationError>,
    /// Non-fatal warnings about best-practice violations.
    pub warnings: Vec<SchemaValidationWarning>,
}

impl ValidationReport {
    /// Create a new empty `ValidationReport` with the given validity flag.
    pub fn new(is_valid: bool) -> Self {
        Self {
            is_valid,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add an error; also marks the report as invalid.
    pub fn add_error(&mut self, error: SchemaValidationError) {
        self.errors.push(error);
        self.is_valid = false;
    }

    /// Add a non-fatal warning.
    pub fn add_warning(&mut self, warning: SchemaValidationWarning) {
        self.warnings.push(warning);
    }

    /// Returns `true` when neither errors nor warnings are present.
    pub fn is_clean(&self) -> bool {
        self.errors.is_empty() && self.warnings.is_empty()
    }
}

/// A single validation error tied to a SAMM element.
#[derive(Debug, Clone)]
pub struct SchemaValidationError {
    /// The URN of the SAMM element that triggered the error.
    pub element_urn: String,
    /// The rule that was violated.
    pub rule: ValidationRule,
    /// Human-readable error message.
    pub message: String,
}

/// A non-fatal validation warning.
#[derive(Debug, Clone)]
pub struct SchemaValidationWarning {
    /// The URN of the SAMM element that triggered the warning.
    pub element_urn: String,
    /// Human-readable warning message.
    pub message: String,
}

/// Rules checked by [`SammSchemaValidator`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationRule {
    /// A Property must declare a Characteristic.
    PropertyMissingCharacteristic,
    /// An Entity must contain at least one Property.
    EntityHasNoProperties,
    /// An Enumeration Characteristic must list at least one value.
    EnumerationEmpty,
    /// A Measurement Characteristic must specify a unit.
    MeasurementMissingUnit,
    /// A Quantifiable Characteristic must specify a unit.
    QuantifiableMissingUnit,
    /// A Collection/List/Set/SortedSet/TimeSeries element type must be valid.
    CollectionInvalidElementType,
    /// A Duration Characteristic must specify a time unit.
    DurationMissingTimeUnit,
    /// A Trait Characteristic that wraps a base characteristic must not be
    /// paired with an incompatible Constraint type.
    IncompatibleConstraint,
    /// Property names must be camelCase.
    InvalidPropertyNaming,
    /// Aspect must have at least one Property.
    AspectHasNoProperties,
    /// Characteristic data type must be a recognised XSD type or entity URN.
    UnrecognisedDataType,
    /// Element URN does not conform to the `urn:samm:…` pattern.
    InvalidUrn,
    /// An Either Characteristic must reference exactly two characteristics.
    EitherMissingBranch,
    /// A custom / user-defined rule.
    Custom(String),
}

impl std::fmt::Display for ValidationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationRule::PropertyMissingCharacteristic => {
                write!(f, "PropertyMissingCharacteristic")
            }
            ValidationRule::EntityHasNoProperties => write!(f, "EntityHasNoProperties"),
            ValidationRule::EnumerationEmpty => write!(f, "EnumerationEmpty"),
            ValidationRule::MeasurementMissingUnit => write!(f, "MeasurementMissingUnit"),
            ValidationRule::QuantifiableMissingUnit => write!(f, "QuantifiableMissingUnit"),
            ValidationRule::CollectionInvalidElementType => {
                write!(f, "CollectionInvalidElementType")
            }
            ValidationRule::DurationMissingTimeUnit => write!(f, "DurationMissingTimeUnit"),
            ValidationRule::IncompatibleConstraint => write!(f, "IncompatibleConstraint"),
            ValidationRule::InvalidPropertyNaming => write!(f, "InvalidPropertyNaming"),
            ValidationRule::AspectHasNoProperties => write!(f, "AspectHasNoProperties"),
            ValidationRule::UnrecognisedDataType => write!(f, "UnrecognisedDataType"),
            ValidationRule::InvalidUrn => write!(f, "InvalidUrn"),
            ValidationRule::EitherMissingBranch => write!(f, "EitherMissingBranch"),
            ValidationRule::Custom(s) => write!(f, "Custom({})", s),
        }
    }
}

/// Validates SAMM Aspect Model definitions against the SAMM 2.3.0 specification.
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_samm::metamodel::Aspect;
/// use oxirs_samm::validation::SammSchemaValidator;
///
/// let aspect = Aspect::new("urn:samm:org.example:1.0.0#Movement".to_string());
/// let validator = SammSchemaValidator::new();
/// let report = validator.validate_aspect(&aspect);
/// if !report.is_valid {
///     for err in &report.errors {
///         eprintln!("[{}] {}", err.rule, err.message);
///     }
/// }
/// ```
#[derive(Debug, Default)]
pub struct SammSchemaValidator {
    /// Whether to enforce strict naming conventions.
    enforce_naming: bool,
    /// Whether to warn about missing preferred names.
    warn_missing_names: bool,
    /// Whether to warn about missing descriptions.
    warn_missing_descriptions: bool,
}

impl SammSchemaValidator {
    /// Create a new validator with all checks enabled.
    pub fn new() -> Self {
        Self {
            enforce_naming: true,
            warn_missing_names: true,
            warn_missing_descriptions: false,
        }
    }

    /// Disable naming-convention checks.
    pub fn without_naming_checks(mut self) -> Self {
        self.enforce_naming = false;
        self
    }

    /// Enable warnings for missing preferred names.
    pub fn with_name_warnings(mut self) -> Self {
        self.warn_missing_names = true;
        self
    }

    /// Enable warnings for missing English descriptions.
    pub fn with_description_warnings(mut self) -> Self {
        self.warn_missing_descriptions = true;
        self
    }

    // ------------------------------------------------------------------ //
    //  Public entry points                                                 //
    // ------------------------------------------------------------------ //

    /// Validate a complete SAMM Aspect model and return a [`ValidationReport`].
    pub fn validate_aspect(&self, aspect: &Aspect) -> ValidationReport {
        let mut report = ValidationReport::new(true);

        // Aspect-level rules
        self.check_urn(aspect.urn(), aspect.urn(), &mut report);
        self.check_aspect_has_properties(aspect, &mut report);

        if self.warn_missing_names && aspect.metadata().preferred_names.is_empty() {
            report.add_warning(SchemaValidationWarning {
                element_urn: aspect.urn().to_string(),
                message: "Aspect should declare at least one preferred name".to_string(),
            });
        }

        // Property-level rules
        for prop in aspect.properties() {
            let prop_errors = self.validate_property(prop);
            for e in prop_errors {
                report.add_error(e);
            }

            // Naming convention (warning, not error, unless enforce_naming=true)
            if let Some(warn_or_err) = self.check_naming(&prop.name(), prop.urn()) {
                if self.enforce_naming {
                    report.add_error(warn_or_err);
                } else {
                    report.add_warning(SchemaValidationWarning {
                        element_urn: prop.urn().to_string(),
                        message: warn_or_err.message,
                    });
                }
            }

            // Characteristic-level rules
            if let Some(char) = &prop.characteristic {
                let char_errors = self.validate_characteristic(char);
                for e in char_errors {
                    report.add_error(e);
                }
            }
        }

        report
    }

    /// Validate a single [`Property`] and return any errors found.
    pub fn validate_property(&self, prop: &Property) -> Vec<SchemaValidationError> {
        let mut errors = Vec::new();

        self.check_urn_into(prop.urn(), prop.urn(), &mut errors);

        if prop.characteristic.is_none() && !prop.is_abstract {
            errors.push(SchemaValidationError {
                element_urn: prop.urn().to_string(),
                rule: ValidationRule::PropertyMissingCharacteristic,
                message: format!(
                    "Property '{}' must declare a Characteristic (samm:characteristic)",
                    prop.name()
                ),
            });
        }

        errors
    }

    /// Validate a single [`Characteristic`] and return any errors found.
    pub fn validate_characteristic(&self, char: &Characteristic) -> Vec<SchemaValidationError> {
        let mut errors = Vec::new();
        let urn = char.urn().to_string();

        self.check_urn_into(&urn, &urn, &mut errors);
        self.check_characteristic_kind(char, &urn, &mut errors);

        // Validate nested / wrapped characteristics recursively
        match char.kind() {
            CharacteristicKind::Collection {
                element_characteristic: Some(inner),
            }
            | CharacteristicKind::List {
                element_characteristic: Some(inner),
            }
            | CharacteristicKind::Set {
                element_characteristic: Some(inner),
            }
            | CharacteristicKind::SortedSet {
                element_characteristic: Some(inner),
            }
            | CharacteristicKind::TimeSeries {
                element_characteristic: Some(inner),
            } => {
                let inner_errors = self.validate_characteristic(inner);
                errors.extend(inner_errors);
            }
            CharacteristicKind::Either { left, right } => {
                errors.extend(self.validate_characteristic(left));
                errors.extend(self.validate_characteristic(right));
            }
            _ => {}
        }

        errors
    }

    /// Validate an [`Entity`] and return any errors found.
    pub fn validate_entity(&self, entity: &Entity) -> Vec<SchemaValidationError> {
        let mut errors = Vec::new();
        let urn = entity.urn().to_string();

        self.check_urn_into(&urn, &urn, &mut errors);

        if entity.properties().is_empty() && !entity.is_abstract {
            errors.push(SchemaValidationError {
                element_urn: urn.clone(),
                rule: ValidationRule::EntityHasNoProperties,
                message: format!("Entity '{}' must have at least one Property", entity.name()),
            });
        }

        for prop in entity.properties() {
            errors.extend(self.validate_property(prop));
            if let Some(char) = &prop.characteristic {
                errors.extend(self.validate_characteristic(char));
            }
        }

        errors
    }

    // ------------------------------------------------------------------ //
    //  Private helpers                                                     //
    // ------------------------------------------------------------------ //

    fn check_aspect_has_properties(&self, aspect: &Aspect, report: &mut ValidationReport) {
        if aspect.properties().is_empty() && aspect.operations().is_empty() {
            report.add_error(SchemaValidationError {
                element_urn: aspect.urn().to_string(),
                rule: ValidationRule::AspectHasNoProperties,
                message: format!(
                    "Aspect '{}' must declare at least one Property or Operation",
                    aspect.name()
                ),
            });
        }
    }

    fn check_urn(&self, urn: &str, element_urn: &str, report: &mut ValidationReport) {
        if !is_valid_samm_urn(urn) {
            report.add_warning(SchemaValidationWarning {
                element_urn: element_urn.to_string(),
                message: format!(
                    "URN '{}' does not follow the recommended 'urn:samm:<namespace>:<version>#<name>' pattern",
                    urn
                ),
            });
        }
    }

    fn check_urn_into(
        &self,
        urn: &str,
        element_urn: &str,
        errors: &mut Vec<SchemaValidationError>,
    ) {
        if urn.is_empty() {
            errors.push(SchemaValidationError {
                element_urn: element_urn.to_string(),
                rule: ValidationRule::InvalidUrn,
                message: "Element URN must not be empty".to_string(),
            });
        }
    }

    /// Returns `Some(error)` when the `name` violates camelCase convention.
    fn check_naming(&self, name: &str, element_urn: &str) -> Option<SchemaValidationError> {
        if name.is_empty() {
            return None;
        }
        if !Self::is_camel_case(name) {
            return Some(SchemaValidationError {
                element_urn: element_urn.to_string(),
                rule: ValidationRule::InvalidPropertyNaming,
                message: format!(
                    "Name '{}' should follow camelCase convention (e.g., 'myProperty')",
                    name
                ),
            });
        }
        None
    }

    fn check_characteristic_kind(
        &self,
        char: &Characteristic,
        urn: &str,
        errors: &mut Vec<SchemaValidationError>,
    ) {
        match char.kind() {
            CharacteristicKind::Enumeration { values } => {
                if values.is_empty() {
                    errors.push(SchemaValidationError {
                        element_urn: urn.to_string(),
                        rule: ValidationRule::EnumerationEmpty,
                        message: format!(
                            "Enumeration '{}' must contain at least one value",
                            char.name()
                        ),
                    });
                }
            }
            CharacteristicKind::State { values, .. } => {
                if values.is_empty() {
                    errors.push(SchemaValidationError {
                        element_urn: urn.to_string(),
                        rule: ValidationRule::EnumerationEmpty,
                        message: format!("State '{}' must contain at least one value", char.name()),
                    });
                }
            }
            CharacteristicKind::Measurement { unit } => {
                if unit.is_empty() {
                    errors.push(SchemaValidationError {
                        element_urn: urn.to_string(),
                        rule: ValidationRule::MeasurementMissingUnit,
                        message: format!(
                            "Measurement '{}' must specify a unit (samm-c:unit)",
                            char.name()
                        ),
                    });
                }
            }
            CharacteristicKind::Quantifiable { unit } => {
                if unit.is_empty() {
                    errors.push(SchemaValidationError {
                        element_urn: urn.to_string(),
                        rule: ValidationRule::QuantifiableMissingUnit,
                        message: format!(
                            "Quantifiable '{}' must specify a unit (samm-c:unit)",
                            char.name()
                        ),
                    });
                }
            }
            CharacteristicKind::Duration { unit } => {
                if unit.is_empty() || !is_time_unit(unit) {
                    errors.push(SchemaValidationError {
                        element_urn: urn.to_string(),
                        rule: ValidationRule::DurationMissingTimeUnit,
                        message: format!(
                            "Duration '{}' must specify a time unit \
                            (e.g., unit:second, unit:millisecond)",
                            char.name()
                        ),
                    });
                }
            }
            CharacteristicKind::Collection {
                element_characteristic,
            }
            | CharacteristicKind::List {
                element_characteristic,
            }
            | CharacteristicKind::Set {
                element_characteristic,
            }
            | CharacteristicKind::SortedSet {
                element_characteristic,
            }
            | CharacteristicKind::TimeSeries {
                element_characteristic,
            } => {
                // If an element characteristic is declared it must not itself
                // be a bare trait without a data type.
                if let Some(inner) = element_characteristic {
                    if inner.data_type.is_none()
                        && matches!(inner.kind(), CharacteristicKind::Trait)
                    {
                        errors.push(SchemaValidationError {
                            element_urn: urn.to_string(),
                            rule: ValidationRule::CollectionInvalidElementType,
                            message: format!(
                                "Collection element characteristic '{}' must declare a data type",
                                inner.name()
                            ),
                        });
                    }
                }
            }
            CharacteristicKind::Either { .. } => {
                // Both branches validated recursively above.
            }
            CharacteristicKind::Trait => {
                // Trait is the default fallback; constraints must be compatible.
                self.check_trait_constraints(char, urn, errors);
            }
            _ => {}
        }

        // Validate data type when present
        if let Some(dt) = &char.data_type {
            if !is_known_data_type(dt) {
                errors.push(SchemaValidationError {
                    element_urn: urn.to_string(),
                    rule: ValidationRule::UnrecognisedDataType,
                    message: format!(
                        "Data type '{}' is not a recognised XSD type or SAMM entity URN",
                        dt
                    ),
                });
            }
        }
    }

    fn check_trait_constraints(
        &self,
        char: &Characteristic,
        urn: &str,
        errors: &mut Vec<SchemaValidationError>,
    ) {
        use crate::metamodel::Constraint;

        for constraint in &char.constraints {
            // Language/locale constraints only make sense on string types
            let is_string_like = char
                .data_type
                .as_deref()
                .map(|dt| dt.ends_with("string") || dt.ends_with("langString"))
                .unwrap_or(true); // allow if unknown

            if !is_string_like {
                match constraint {
                    Constraint::LanguageConstraint { .. } | Constraint::LocaleConstraint { .. } => {
                        errors.push(SchemaValidationError {
                            element_urn: urn.to_string(),
                            rule: ValidationRule::IncompatibleConstraint,
                            message: format!(
                                "Language/Locale constraint on '{}' is only valid for \
                                string-like data types",
                                char.name()
                            ),
                        });
                    }
                    _ => {}
                }
            }
        }
    }

    // ------------------------------------------------------------------ //
    //  Naming utilities (pub for external reuse)                          //
    // ------------------------------------------------------------------ //

    /// Returns `true` if `s` is valid camelCase (starts with lowercase, no
    /// underscores, no spaces, not empty).
    pub fn is_camel_case(s: &str) -> bool {
        if s.is_empty() {
            return false;
        }
        let mut chars = s.chars();
        let first = chars.next().expect("non-empty string has first char");
        if !first.is_lowercase() {
            return false;
        }
        // Must not contain underscores or spaces
        !s.contains('_') && !s.contains(' ') && !s.contains('-')
    }

    /// Returns `true` if `s` is valid PascalCase (starts with uppercase, no
    /// underscores, no spaces, not empty).
    pub fn is_pascal_case(s: &str) -> bool {
        if s.is_empty() {
            return false;
        }
        let mut chars = s.chars();
        let first = chars.next().expect("non-empty string has first char");
        if !first.is_uppercase() {
            return false;
        }
        !s.contains('_') && !s.contains(' ') && !s.contains('-')
    }
}

// ------------------------------------------------------------------ //
//  Stand-alone helper functions                                        //
// ------------------------------------------------------------------ //

/// Validate that a URN starts with `urn:samm:` and contains a `#` separator.
fn is_valid_samm_urn(urn: &str) -> bool {
    urn.starts_with("urn:samm:") && urn.contains('#')
}

/// Return `true` if the unit string looks like a SAMM time unit.
fn is_time_unit(unit: &str) -> bool {
    const TIME_UNITS: &[&str] = &[
        "unit:second",
        "unit:millisecond",
        "unit:microsecond",
        "unit:nanosecond",
        "unit:minute",
        "unit:hour",
        "unit:day",
        "unit:week",
        "unit:month",
        "unit:year",
        // Allow partial matches (e.g. full URN forms)
        "second",
        "millisecond",
        "microsecond",
        "nanosecond",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
    ];
    TIME_UNITS
        .iter()
        .any(|&t| unit.ends_with(t) || unit.contains(t))
}

/// Return `true` if the data-type string is a recognised XSD type or a
/// `urn:samm:` / `urn:` entity reference.
fn is_known_data_type(dt: &str) -> bool {
    const XSD_BASE: &str = "http://www.w3.org/2001/XMLSchema#";
    const XSD_TYPES: &[&str] = &[
        "string",
        "boolean",
        "decimal",
        "float",
        "double",
        "integer",
        "int",
        "long",
        "short",
        "byte",
        "unsignedInt",
        "unsignedLong",
        "unsignedShort",
        "unsignedByte",
        "positiveInteger",
        "negativeInteger",
        "nonNegativeInteger",
        "nonPositiveInteger",
        "date",
        "time",
        "dateTime",
        "dateTimeStamp",
        "duration",
        "gYear",
        "gYearMonth",
        "gMonth",
        "gMonthDay",
        "gDay",
        "anyURI",
        "base64Binary",
        "hexBinary",
        "langString",
        "curie",
    ];

    // Full XSD URL format
    if dt.starts_with(XSD_BASE) {
        let local = dt.trim_start_matches(XSD_BASE);
        return XSD_TYPES.contains(&local);
    }
    // Prefixed XSD format (e.g. xsd:string)
    if dt.starts_with("xsd:") {
        let local = dt.trim_start_matches("xsd:");
        return XSD_TYPES.contains(&local);
    }
    // Entity URN references
    if dt.starts_with("urn:") {
        return true;
    }
    // RDF / RDFS types
    if dt.starts_with("rdf:") || dt.starts_with("rdfs:") {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Aspect, Characteristic, CharacteristicKind, Property};

    // Helper: build a minimal valid aspect
    fn valid_aspect() -> Aspect {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#Movement".to_string());
        aspect
            .metadata
            .add_preferred_name("en".to_string(), "Movement".to_string());

        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#SpeedChar".to_string(),
            CharacteristicKind::Measurement {
                unit: "unit:kilometrePerHour".to_string(),
            },
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string());

        let prop =
            Property::new("urn:samm:org.example:1.0.0#speed".to_string()).with_characteristic(char);

        aspect.add_property(prop);
        aspect
    }

    #[test]
    fn test_valid_aspect_passes() {
        let aspect = valid_aspect();
        let validator = SammSchemaValidator::new();
        let report = validator.validate_aspect(&aspect);
        assert!(report.is_valid, "errors: {:?}", report.errors);
    }

    #[test]
    fn test_aspect_no_properties_fails() {
        let aspect = Aspect::new("urn:samm:org.example:1.0.0#Empty".to_string());
        let validator = SammSchemaValidator::new();
        let report = validator.validate_aspect(&aspect);
        assert!(!report.is_valid);
        assert!(report
            .errors
            .iter()
            .any(|e| e.rule == ValidationRule::AspectHasNoProperties));
    }

    #[test]
    fn test_property_missing_characteristic() {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
        let prop = Property::new("urn:samm:org.example:1.0.0#speed".to_string());
        aspect.add_property(prop);

        let validator = SammSchemaValidator::new();
        let report = validator.validate_aspect(&aspect);
        assert!(!report.is_valid);
        assert!(report
            .errors
            .iter()
            .any(|e| e.rule == ValidationRule::PropertyMissingCharacteristic));
    }

    #[test]
    fn test_enumeration_empty_fails() {
        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#StatusEnum".to_string(),
            CharacteristicKind::Enumeration { values: vec![] },
        );

        let validator = SammSchemaValidator::new();
        let errors = validator.validate_characteristic(&char);
        assert!(errors
            .iter()
            .any(|e| e.rule == ValidationRule::EnumerationEmpty));
    }

    #[test]
    fn test_enumeration_non_empty_passes() {
        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#StatusEnum".to_string(),
            CharacteristicKind::Enumeration {
                values: vec!["Active".to_string(), "Inactive".to_string()],
            },
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string());

        let validator = SammSchemaValidator::new();
        let errors = validator.validate_characteristic(&char);
        assert!(
            errors
                .iter()
                .all(|e| e.rule != ValidationRule::EnumerationEmpty),
            "unexpected errors: {:?}",
            errors
        );
    }

    #[test]
    fn test_measurement_missing_unit() {
        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#Speed".to_string(),
            CharacteristicKind::Measurement {
                unit: String::new(),
            },
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string());

        let validator = SammSchemaValidator::new();
        let errors = validator.validate_characteristic(&char);
        assert!(errors
            .iter()
            .any(|e| e.rule == ValidationRule::MeasurementMissingUnit));
    }

    #[test]
    fn test_duration_missing_time_unit() {
        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#Duration".to_string(),
            CharacteristicKind::Duration {
                unit: "unit:metre".to_string(),
            },
        );

        let validator = SammSchemaValidator::new();
        let errors = validator.validate_characteristic(&char);
        assert!(errors
            .iter()
            .any(|e| e.rule == ValidationRule::DurationMissingTimeUnit));
    }

    #[test]
    fn test_duration_with_valid_time_unit() {
        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#Duration".to_string(),
            CharacteristicKind::Duration {
                unit: "unit:second".to_string(),
            },
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#decimal".to_string());

        let validator = SammSchemaValidator::new();
        let errors = validator.validate_characteristic(&char);
        assert!(
            errors
                .iter()
                .all(|e| e.rule != ValidationRule::DurationMissingTimeUnit),
            "unexpected errors: {:?}",
            errors
        );
    }

    #[test]
    fn test_invalid_property_naming() {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#SpeedChar".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string());
        // Property name starts with uppercase -> violates camelCase
        let prop =
            Property::new("urn:samm:org.example:1.0.0#Speed".to_string()).with_characteristic(char);
        aspect.add_property(prop);

        let validator = SammSchemaValidator::new();
        let report = validator.validate_aspect(&aspect);
        assert!(report
            .errors
            .iter()
            .any(|e| e.rule == ValidationRule::InvalidPropertyNaming));
    }

    #[test]
    fn test_camel_case_detection() {
        assert!(SammSchemaValidator::is_camel_case("speed"));
        assert!(SammSchemaValidator::is_camel_case("speedValue"));
        assert!(SammSchemaValidator::is_camel_case("myLongPropertyName"));
        assert!(!SammSchemaValidator::is_camel_case("Speed"));
        assert!(!SammSchemaValidator::is_camel_case("speed_value"));
        assert!(!SammSchemaValidator::is_camel_case(""));
    }

    #[test]
    fn test_pascal_case_detection() {
        assert!(SammSchemaValidator::is_pascal_case("Speed"));
        assert!(SammSchemaValidator::is_pascal_case("SpeedValue"));
        assert!(SammSchemaValidator::is_pascal_case("MyEntity"));
        assert!(!SammSchemaValidator::is_pascal_case("speed"));
        assert!(!SammSchemaValidator::is_pascal_case("Speed_Value"));
        assert!(!SammSchemaValidator::is_pascal_case(""));
    }

    #[test]
    fn test_unrecognised_data_type() {
        let char = Characteristic::new(
            "urn:samm:org.example:1.0.0#Char".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("completely_unknown_type".to_string());

        let validator = SammSchemaValidator::new();
        let errors = validator.validate_characteristic(&char);
        assert!(errors
            .iter()
            .any(|e| e.rule == ValidationRule::UnrecognisedDataType));
    }

    #[test]
    fn test_entity_no_properties_fails() {
        let entity = Entity::new("urn:samm:org.example:1.0.0#EmptyEntity".to_string());
        let validator = SammSchemaValidator::new();
        let errors = validator.validate_entity(&entity);
        assert!(errors
            .iter()
            .any(|e| e.rule == ValidationRule::EntityHasNoProperties));
    }

    #[test]
    fn test_abstract_entity_no_properties_ok() {
        let entity =
            Entity::new("urn:samm:org.example:1.0.0#AbstractBase".to_string()).as_abstract();
        let validator = SammSchemaValidator::new();
        let errors = validator.validate_entity(&entity);
        assert!(
            errors
                .iter()
                .all(|e| e.rule != ValidationRule::EntityHasNoProperties),
            "abstract entities may have no properties"
        );
    }

    #[test]
    fn test_validation_report_add_error_marks_invalid() {
        let mut report = ValidationReport::new(true);
        assert!(report.is_valid);
        report.add_error(SchemaValidationError {
            element_urn: "urn:test".to_string(),
            rule: ValidationRule::AspectHasNoProperties,
            message: "test".to_string(),
        });
        assert!(!report.is_valid);
        assert!(!report.is_clean());
    }
}
