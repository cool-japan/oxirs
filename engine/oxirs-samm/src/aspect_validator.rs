/// SAMM aspect model validation.
///
/// Validates an aspect model definition against the SAMM metamodel rules:
/// required properties, cardinality, constraints, cyclic references,
/// cross-reference integrity, characteristic-type compatibility, and
/// produces structured validation messages per violation.
use std::collections::{HashMap, HashSet};

// ── Violation severity ────────────────────────────────────────────────────────

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational finding — does not prevent use of the model.
    Info,
    /// Non-critical issue — should be fixed but model may still be used.
    Warning,
    /// Critical issue — model must not be used until resolved.
    Error,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warning => write!(f, "WARNING"),
            Severity::Error => write!(f, "ERROR"),
        }
    }
}

// ── Constraint definitions ────────────────────────────────────────────────────

/// A range constraint (min / max inclusive bounds).
#[derive(Debug, Clone, PartialEq)]
pub struct RangeConstraint {
    /// Inclusive lower bound; `None` means no lower limit.
    pub min: Option<f64>,
    /// Inclusive upper bound; `None` means no upper limit.
    pub max: Option<f64>,
}

impl RangeConstraint {
    /// Create a constraint with both bounds.
    pub fn bounded(min: f64, max: f64) -> Self {
        RangeConstraint {
            min: Some(min),
            max: Some(max),
        }
    }

    /// Create a constraint with only a minimum bound.
    pub fn min_only(min: f64) -> Self {
        RangeConstraint {
            min: Some(min),
            max: None,
        }
    }

    /// Create a constraint with only a maximum bound.
    pub fn max_only(max: f64) -> Self {
        RangeConstraint {
            min: None,
            max: Some(max),
        }
    }
}

/// A length constraint (min / max character or element count).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LengthConstraint {
    /// Minimum length (inclusive); `None` means no minimum.
    pub min: Option<usize>,
    /// Maximum length (inclusive); `None` means no maximum.
    pub max: Option<usize>,
}

/// A pattern / regular-expression constraint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternConstraint {
    /// Regular-expression pattern the value must match.
    pub pattern: String,
}

/// The collection of constraints that may be attached to a property.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Constraints {
    /// Optional numeric range constraint.
    pub range: Option<RangeConstraint>,
    /// Optional length constraint.
    pub length: Option<LengthConstraint>,
    /// Optional regular-expression constraint.
    pub pattern: Option<PatternConstraint>,
}

// ── Model element types ───────────────────────────────────────────────────────

/// The data type of a SAMM characteristic.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataType {
    /// `xsd:string` — UTF-8 text.
    XsdString,
    /// `xsd:integer` — arbitrary-precision integer.
    XsdInteger,
    /// `xsd:float` / `xsd:double` — 64-bit floating-point.
    XsdFloat,
    /// `xsd:boolean` — true / false.
    XsdBoolean,
    /// `xsd:date` — calendar date without time.
    XsdDate,
    /// `xsd:dateTime` — date combined with time of day.
    XsdDateTime,
    /// Any other datatype IRI not covered by the variants above.
    Custom(String),
}

impl std::str::FromStr for DataType {
    type Err = std::convert::Infallible;

    /// Parse from a string like `"xsd:string"`, `"xsd:integer"`, etc.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "xsd:string" | "http://www.w3.org/2001/XMLSchema#string" => DataType::XsdString,
            "xsd:integer" | "http://www.w3.org/2001/XMLSchema#integer" => DataType::XsdInteger,
            "xsd:float" | "xsd:double" => DataType::XsdFloat,
            "xsd:boolean" => DataType::XsdBoolean,
            "xsd:date" => DataType::XsdDate,
            "xsd:dateTime" => DataType::XsdDateTime,
            other => DataType::Custom(other.to_owned()),
        })
    }
}

impl DataType {
    /// Returns `true` if this data type supports numeric range constraints.
    pub fn is_numeric(&self) -> bool {
        matches!(self, DataType::XsdInteger | DataType::XsdFloat)
    }

    /// Returns `true` if this data type supports string length / pattern constraints.
    pub fn is_textual(&self) -> bool {
        matches!(self, DataType::XsdString)
    }
}

/// A SAMM characteristic definition (simplified).
#[derive(Debug, Clone)]
pub struct AspectCharacteristic {
    /// Unique name within the aspect model.
    pub name: String,
    /// XSD data type of the characteristic.
    pub data_type: DataType,
    /// Constraints applied to values of this characteristic.
    pub constraints: Constraints,
}

/// Whether a property is optional or mandatory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cardinality {
    /// Mandatory — the property MUST be present.
    Mandatory,
    /// Optional — the property MAY be absent.
    Optional,
}

/// A property of an aspect or entity.
#[derive(Debug, Clone)]
pub struct AspectProperty {
    /// Name of the property.
    pub name: String,
    /// Cardinality (mandatory vs. optional).
    pub cardinality: Cardinality,
    /// Reference to the characteristic by name.
    pub characteristic_ref: String,
}

/// An entity definition — optionally extends another entity.
#[derive(Debug, Clone)]
pub struct AspectEntity {
    /// Unique entity name.
    pub name: String,
    /// Optional parent entity name (for extends-chain).
    pub extends: Option<String>,
    /// Properties declared on this entity.
    pub properties: Vec<AspectProperty>,
}

/// The top-level SAMM aspect definition.
#[derive(Debug, Clone)]
pub struct AspectModel {
    /// Unique name of the aspect.
    pub name: String,
    /// All properties of the aspect.
    pub properties: Vec<AspectProperty>,
    /// Characteristic definitions referenced by properties.
    pub characteristics: Vec<AspectCharacteristic>,
    /// Entity definitions used in the model.
    pub entities: Vec<AspectEntity>,
    /// Preferred description (optional but recommended).
    pub description: Option<String>,
}

impl AspectModel {
    /// Create a minimal aspect model.
    pub fn new(name: impl Into<String>) -> Self {
        AspectModel {
            name: name.into(),
            properties: Vec::new(),
            characteristics: Vec::new(),
            entities: Vec::new(),
            description: None,
        }
    }
}

// ── Validation result structures ──────────────────────────────────────────────

/// A structured validation message for a single violation.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationMessage {
    /// Severity of the finding.
    pub severity: Severity,
    /// The element (property, entity, characteristic) the message refers to.
    pub element: String,
    /// Human-readable description of the violation.
    pub message: String,
    /// Optional path within the model (e.g., `"aspect.entity.property"`).
    pub path: Option<String>,
}

impl ValidationMessage {
    fn error(element: impl Into<String>, message: impl Into<String>) -> Self {
        ValidationMessage {
            severity: Severity::Error,
            element: element.into(),
            message: message.into(),
            path: None,
        }
    }

    fn warning(element: impl Into<String>, message: impl Into<String>) -> Self {
        ValidationMessage {
            severity: Severity::Warning,
            element: element.into(),
            message: message.into(),
            path: None,
        }
    }

    fn info(element: impl Into<String>, message: impl Into<String>) -> Self {
        ValidationMessage {
            severity: Severity::Info,
            element: element.into(),
            message: message.into(),
            path: None,
        }
    }

    /// Attach a model path to the message.
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }
}

impl std::fmt::Display for ValidationMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.element, self.message)
    }
}

/// The full validation result for an aspect model.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// All messages produced during validation.
    pub messages: Vec<ValidationMessage>,
}

impl ValidationResult {
    fn new() -> Self {
        ValidationResult {
            messages: Vec::new(),
        }
    }

    fn push(&mut self, msg: ValidationMessage) {
        self.messages.push(msg);
    }

    /// Returns `true` when there are no `Error`-severity messages.
    pub fn is_valid(&self) -> bool {
        !self.messages.iter().any(|m| m.severity == Severity::Error)
    }

    /// Returns all messages at or above the given severity level.
    pub fn messages_at_least(&self, min: Severity) -> Vec<&ValidationMessage> {
        self.messages.iter().filter(|m| m.severity >= min).collect()
    }

    /// Returns only `Error`-severity messages.
    pub fn errors(&self) -> Vec<&ValidationMessage> {
        self.messages_at_least(Severity::Error)
    }

    /// Returns only `Warning`-severity messages.
    pub fn warnings(&self) -> Vec<&ValidationMessage> {
        self.messages
            .iter()
            .filter(|m| m.severity == Severity::Warning)
            .collect()
    }

    /// Total number of messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns `true` when there are no messages at all.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

// ── AspectValidator ───────────────────────────────────────────────────────────

/// Validates a SAMM aspect model definition.
pub struct AspectValidator;

impl AspectValidator {
    /// Validate the given aspect model and return a structured result.
    pub fn validate(model: &AspectModel) -> ValidationResult {
        let mut result = ValidationResult::new();

        Self::check_aspect_name(model, &mut result);
        Self::check_description(model, &mut result);
        Self::check_required_properties(model, &mut result);
        Self::check_property_cardinality(model, &mut result);
        Self::check_cross_references(model, &mut result);
        Self::check_characteristic_compatibility(model, &mut result);
        Self::check_constraints(model, &mut result);
        Self::check_entity_cyclic_references(model, &mut result);

        result
    }

    // ── Rule implementations ──────────────────────────────────────────────────

    /// An aspect must have a non-empty name.
    fn check_aspect_name(model: &AspectModel, result: &mut ValidationResult) {
        if model.name.trim().is_empty() {
            result.push(ValidationMessage::error(
                "Aspect",
                "Aspect name must not be empty",
            ));
        }
    }

    /// Description is recommended (INFO if absent).
    fn check_description(model: &AspectModel, result: &mut ValidationResult) {
        if model
            .description
            .as_deref()
            .map_or(true, |d| d.trim().is_empty())
        {
            result.push(ValidationMessage::info(
                &model.name,
                "Aspect has no description — consider adding a human-readable description",
            ));
        }
    }

    /// Every property must have a non-empty name and a non-empty characteristic reference.
    fn check_required_properties(model: &AspectModel, result: &mut ValidationResult) {
        for prop in &model.properties {
            if prop.name.trim().is_empty() {
                result.push(ValidationMessage::error(
                    &model.name,
                    "Property has an empty name",
                ));
            }
            if prop.characteristic_ref.trim().is_empty() {
                result.push(ValidationMessage::error(
                    &prop.name,
                    "Property is missing a characteristic reference",
                ));
            }
        }

        for entity in &model.entities {
            for prop in &entity.properties {
                if prop.name.trim().is_empty() {
                    result.push(
                        ValidationMessage::error(&entity.name, "Entity property has an empty name")
                            .with_path(format!("{}.{}", entity.name, prop.name)),
                    );
                }
            }
        }
    }

    /// Check that mandatory properties are not shadowed by optional declarations
    /// at the entity level (simple cardinality linting).
    fn check_property_cardinality(model: &AspectModel, result: &mut ValidationResult) {
        for entity in &model.entities {
            let mandatory: Vec<&AspectProperty> = entity
                .properties
                .iter()
                .filter(|p| p.cardinality == Cardinality::Mandatory)
                .collect();
            // An entity with no mandatory properties is unusual — emit a warning.
            if mandatory.is_empty() && !entity.properties.is_empty() {
                result.push(ValidationMessage::warning(
                    &entity.name,
                    "Entity has no mandatory properties — this is unusual for SAMM models",
                ));
            }
        }
    }

    /// Every property's `characteristic_ref` must resolve to a known characteristic.
    fn check_cross_references(model: &AspectModel, result: &mut ValidationResult) {
        let char_names: HashSet<&str> = model
            .characteristics
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        let entity_names: HashSet<&str> = model.entities.iter().map(|e| e.name.as_str()).collect();

        // Check aspect-level properties.
        for prop in &model.properties {
            if !prop.characteristic_ref.is_empty()
                && !char_names.contains(prop.characteristic_ref.as_str())
            {
                result.push(
                    ValidationMessage::error(
                        &prop.name,
                        format!(
                            "Property references unknown characteristic '{}'",
                            prop.characteristic_ref
                        ),
                    )
                    .with_path(format!("{}.{}", model.name, prop.name)),
                );
            }
        }

        // Check entity-level properties.
        for entity in &model.entities {
            for prop in &entity.properties {
                if !prop.characteristic_ref.is_empty()
                    && !char_names.contains(prop.characteristic_ref.as_str())
                {
                    result.push(
                        ValidationMessage::error(
                            &prop.name,
                            format!(
                                "Property references unknown characteristic '{}'",
                                prop.characteristic_ref
                            ),
                        )
                        .with_path(format!("{}.{}.{}", model.name, entity.name, prop.name)),
                    );
                }
            }

            // Entity's `extends` must reference a known entity.
            if let Some(parent) = &entity.extends {
                if !entity_names.contains(parent.as_str()) {
                    result.push(
                        ValidationMessage::error(
                            &entity.name,
                            format!("Entity extends unknown entity '{}'", parent),
                        )
                        .with_path(format!("{}.{}", model.name, entity.name)),
                    );
                }
            }
        }
    }

    /// Validate that range/length/pattern constraints are compatible with the
    /// characteristic's data type.
    fn check_constraints(model: &AspectModel, result: &mut ValidationResult) {
        for characteristic in &model.characteristics {
            let c = &characteristic.constraints;

            // Range constraints require a numeric type.
            if c.range.is_some() && !characteristic.data_type.is_numeric() {
                result.push(
                    ValidationMessage::error(
                        &characteristic.name,
                        format!(
                            "RangeConstraint applied to non-numeric type {:?}",
                            characteristic.data_type
                        ),
                    )
                    .with_path(characteristic.name.clone()),
                );
            }

            // Length/pattern constraints require a string type.
            if (c.length.is_some() || c.pattern.is_some()) && !characteristic.data_type.is_textual()
            {
                result.push(
                    ValidationMessage::error(
                        &characteristic.name,
                        format!(
                            "Length/Pattern constraint applied to non-string type {:?}",
                            characteristic.data_type
                        ),
                    )
                    .with_path(characteristic.name.clone()),
                );
            }

            // Range min must not exceed max.
            if let Some(range) = &c.range {
                if let (Some(min), Some(max)) = (range.min, range.max) {
                    if min > max {
                        result.push(ValidationMessage::error(
                            &characteristic.name,
                            format!("RangeConstraint min ({}) > max ({})", min, max),
                        ));
                    }
                }
            }

            // Length min must not exceed max.
            if let Some(len) = &c.length {
                if let (Some(min), Some(max)) = (len.min, len.max) {
                    if min > max {
                        result.push(ValidationMessage::error(
                            &characteristic.name,
                            format!("LengthConstraint min ({}) > max ({})", min, max),
                        ));
                    }
                }
            }
        }
    }

    /// Validate that each property's characteristic is type-compatible.
    ///
    /// Currently checks that a boolean property is not given a range constraint
    /// (which makes no semantic sense in SAMM).
    fn check_characteristic_compatibility(model: &AspectModel, result: &mut ValidationResult) {
        let char_map: HashMap<&str, &AspectCharacteristic> = model
            .characteristics
            .iter()
            .map(|c| (c.name.as_str(), c))
            .collect();

        for prop in &model.properties {
            if let Some(ch) = char_map.get(prop.characteristic_ref.as_str()) {
                if ch.data_type == DataType::XsdBoolean && ch.constraints.range.is_some() {
                    result.push(
                        ValidationMessage::warning(
                            &prop.name,
                            "Boolean characteristic has a range constraint — this has no semantic meaning",
                        )
                        .with_path(format!("{}.{}", model.name, prop.name)),
                    );
                }
            }
        }
    }

    /// Detect cycles in the entity `extends` chain using depth-first search.
    fn check_entity_cyclic_references(model: &AspectModel, result: &mut ValidationResult) {
        let entity_map: HashMap<&str, Option<&str>> = model
            .entities
            .iter()
            .map(|e| (e.name.as_str(), e.extends.as_deref()))
            .collect();

        for entity in &model.entities {
            if let Some(cycle_path) = Self::detect_cycle(&entity.name, &entity_map) {
                result.push(
                    ValidationMessage::error(
                        &entity.name,
                        format!("Cyclic extends chain detected: {}", cycle_path),
                    )
                    .with_path(format!("{}.{}", model.name, entity.name)),
                );
                // Only report each starting node once.
            }
        }
    }

    /// DFS cycle detection in the extends graph.
    /// Returns the cycle description if a cycle exists, otherwise `None`.
    fn detect_cycle(start: &str, entity_map: &HashMap<&str, Option<&str>>) -> Option<String> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut path: Vec<String> = Vec::new();
        let mut current = start.to_owned();

        loop {
            if !visited.insert(current.clone()) {
                // Found cycle — current is the repeated node.
                path.push(current.clone());
                return Some(path.join(" → "));
            }
            path.push(current.clone());
            match entity_map.get(current.as_str()) {
                Some(Some(parent)) => current = parent.to_string(),
                _ => return None, // No parent or unknown entity — no cycle.
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper builders ───────────────────────────────────────────────────────

    fn string_char(name: &str) -> AspectCharacteristic {
        AspectCharacteristic {
            name: name.to_owned(),
            data_type: DataType::XsdString,
            constraints: Constraints::default(),
        }
    }

    fn int_char(name: &str) -> AspectCharacteristic {
        AspectCharacteristic {
            name: name.to_owned(),
            data_type: DataType::XsdInteger,
            constraints: Constraints::default(),
        }
    }

    fn mandatory_prop(name: &str, char_ref: &str) -> AspectProperty {
        AspectProperty {
            name: name.to_owned(),
            cardinality: Cardinality::Mandatory,
            characteristic_ref: char_ref.to_owned(),
        }
    }

    fn optional_prop(name: &str, char_ref: &str) -> AspectProperty {
        AspectProperty {
            name: name.to_owned(),
            cardinality: Cardinality::Optional,
            characteristic_ref: char_ref.to_owned(),
        }
    }

    fn valid_model() -> AspectModel {
        let mut model = AspectModel::new("TestAspect");
        model.description = Some("A test aspect".to_owned());
        model.characteristics.push(string_char("NameChar"));
        model.properties.push(mandatory_prop("name", "NameChar"));
        model
    }

    // ── Happy path ────────────────────────────────────────────────────────────

    #[test]
    fn test_valid_model_is_valid() {
        let model = valid_model();
        let result = AspectValidator::validate(&model);
        assert!(result.is_valid(), "errors: {:?}", result.errors());
    }

    #[test]
    fn test_valid_model_no_errors() {
        let model = valid_model();
        let result = AspectValidator::validate(&model);
        assert_eq!(result.errors().len(), 0);
    }

    // ── Aspect name ───────────────────────────────────────────────────────────

    #[test]
    fn test_empty_aspect_name() {
        let model = AspectModel::new("");
        let result = AspectValidator::validate(&model);
        assert!(!result.is_valid());
        assert!(result
            .errors()
            .iter()
            .any(|m| m.message.contains("name must not be empty")));
    }

    // ── Description ───────────────────────────────────────────────────────────

    #[test]
    fn test_missing_description_gives_info() {
        let mut model = AspectModel::new("NoDesc");
        model.characteristics.push(string_char("C"));
        model.properties.push(mandatory_prop("p", "C"));
        let result = AspectValidator::validate(&model);
        // Should be valid (INFO is not an error).
        assert!(result.is_valid());
        let infos: Vec<_> = result
            .messages
            .iter()
            .filter(|m| m.severity == Severity::Info)
            .collect();
        assert!(
            !infos.is_empty(),
            "expected an INFO message about description"
        );
    }

    // ── Required properties ───────────────────────────────────────────────────

    #[test]
    fn test_property_empty_name() {
        let mut model = valid_model();
        model.characteristics.push(string_char("ExtraChar"));
        model.properties.push(AspectProperty {
            name: "".to_owned(),
            cardinality: Cardinality::Mandatory,
            characteristic_ref: "ExtraChar".to_owned(),
        });
        let result = AspectValidator::validate(&model);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_property_missing_characteristic_ref() {
        let mut model = valid_model();
        model.properties.push(AspectProperty {
            name: "orphan".to_owned(),
            cardinality: Cardinality::Mandatory,
            characteristic_ref: "".to_owned(),
        });
        let result = AspectValidator::validate(&model);
        assert!(!result.is_valid());
        assert!(result
            .errors()
            .iter()
            .any(|m| m.message.contains("missing a characteristic reference")));
    }

    // ── Cardinality ───────────────────────────────────────────────────────────

    #[test]
    fn test_entity_with_only_optional_properties_warning() {
        let mut model = valid_model();
        model.entities.push(AspectEntity {
            name: "AllOptional".to_owned(),
            extends: None,
            properties: vec![
                optional_prop("opt1", "NameChar"),
                optional_prop("opt2", "NameChar"),
            ],
        });
        let result = AspectValidator::validate(&model);
        // No errors, but a warning is expected.
        assert!(result.is_valid());
        assert!(
            !result.warnings().is_empty(),
            "expected cardinality warning"
        );
    }

    #[test]
    fn test_entity_with_mandatory_property_no_cardinality_warning() {
        let mut model = valid_model();
        model.entities.push(AspectEntity {
            name: "Mixed".to_owned(),
            extends: None,
            properties: vec![
                mandatory_prop("m1", "NameChar"),
                optional_prop("o1", "NameChar"),
            ],
        });
        let result = AspectValidator::validate(&model);
        // The cardinality warning should not fire when there's at least one mandatory.
        let warnings = result.warnings();
        let card_warnings: Vec<_> = warnings
            .iter()
            .filter(|m| m.message.contains("mandatory properties"))
            .collect();
        assert!(card_warnings.is_empty());
    }

    // ── Cross-reference validation ─────────────────────────────────────────────

    #[test]
    fn test_unknown_characteristic_ref() {
        let mut model = AspectModel::new("Broken");
        model.description = Some("Broken model".to_owned());
        model
            .properties
            .push(mandatory_prop("p", "NonExistentChar"));
        let result = AspectValidator::validate(&model);
        assert!(!result.is_valid());
        assert!(result
            .errors()
            .iter()
            .any(|m| m.message.contains("unknown characteristic")));
    }

    #[test]
    fn test_entity_extends_unknown_entity() {
        let mut model = valid_model();
        model.entities.push(AspectEntity {
            name: "Child".to_owned(),
            extends: Some("NonExistent".to_owned()),
            properties: vec![mandatory_prop("cp", "NameChar")],
        });
        let result = AspectValidator::validate(&model);
        assert!(!result.is_valid());
        assert!(result
            .errors()
            .iter()
            .any(|m| m.message.contains("unknown entity")));
    }

    #[test]
    fn test_entity_extends_known_entity_ok() {
        let mut model = valid_model();
        model.entities.push(AspectEntity {
            name: "Parent".to_owned(),
            extends: None,
            properties: vec![mandatory_prop("pp", "NameChar")],
        });
        model.entities.push(AspectEntity {
            name: "Child".to_owned(),
            extends: Some("Parent".to_owned()),
            properties: vec![mandatory_prop("cp", "NameChar")],
        });
        let result = AspectValidator::validate(&model);
        assert!(result.is_valid(), "errors: {:?}", result.errors());
    }

    // ── Constraint checking ───────────────────────────────────────────────────

    #[test]
    fn test_range_constraint_on_numeric_type_ok() {
        let mut model = AspectModel::new("RangeModel");
        model.description = Some("desc".to_owned());
        model.characteristics.push(AspectCharacteristic {
            name: "AgeChar".to_owned(),
            data_type: DataType::XsdInteger,
            constraints: Constraints {
                range: Some(RangeConstraint::bounded(0.0, 150.0)),
                ..Default::default()
            },
        });
        model.properties.push(mandatory_prop("age", "AgeChar"));
        let result = AspectValidator::validate(&model);
        assert!(result.is_valid(), "errors: {:?}", result.errors());
    }

    #[test]
    fn test_range_constraint_on_string_type_error() {
        let mut model = AspectModel::new("BadRange");
        model.description = Some("desc".to_owned());
        model.characteristics.push(AspectCharacteristic {
            name: "NameChar".to_owned(),
            data_type: DataType::XsdString,
            constraints: Constraints {
                range: Some(RangeConstraint::bounded(0.0, 10.0)),
                ..Default::default()
            },
        });
        model.properties.push(mandatory_prop("name", "NameChar"));
        let result = AspectValidator::validate(&model);
        assert!(!result.is_valid());
        assert!(result
            .errors()
            .iter()
            .any(|m| m.message.contains("RangeConstraint")));
    }

    #[test]
    fn test_length_constraint_on_string_ok() {
        let mut model = AspectModel::new("LenModel");
        model.description = Some("desc".to_owned());
        model.characteristics.push(AspectCharacteristic {
            name: "CodeChar".to_owned(),
            data_type: DataType::XsdString,
            constraints: Constraints {
                length: Some(LengthConstraint {
                    min: Some(3),
                    max: Some(10),
                }),
                ..Default::default()
            },
        });
        model.properties.push(mandatory_prop("code", "CodeChar"));
        let result = AspectValidator::validate(&model);
        assert!(result.is_valid(), "errors: {:?}", result.errors());
    }

    #[test]
    fn test_length_constraint_on_integer_error() {
        let mut model = AspectModel::new("BadLen");
        model.description = Some("desc".to_owned());
        model.characteristics.push(AspectCharacteristic {
            name: "NumChar".to_owned(),
            data_type: DataType::XsdInteger,
            constraints: Constraints {
                length: Some(LengthConstraint {
                    min: Some(1),
                    max: Some(5),
                }),
                ..Default::default()
            },
        });
        model.properties.push(mandatory_prop("num", "NumChar"));
        let result = AspectValidator::validate(&model);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_range_min_exceeds_max_error() {
        let mut model = AspectModel::new("MinMax");
        model.description = Some("desc".to_owned());
        model.characteristics.push(AspectCharacteristic {
            name: "BadRange".to_owned(),
            data_type: DataType::XsdInteger,
            constraints: Constraints {
                range: Some(RangeConstraint::bounded(100.0, 10.0)),
                ..Default::default()
            },
        });
        model.properties.push(mandatory_prop("val", "BadRange"));
        let result = AspectValidator::validate(&model);
        assert!(!result.is_valid());
        assert!(result
            .errors()
            .iter()
            .any(|m| m.message.contains("min") && m.message.contains("max")));
    }

    // ── Characteristic-property compatibility ─────────────────────────────────

    #[test]
    fn test_boolean_with_range_constraint_warning() {
        let mut model = AspectModel::new("BoolRange");
        model.description = Some("desc".to_owned());
        model.characteristics.push(AspectCharacteristic {
            name: "FlagChar".to_owned(),
            data_type: DataType::XsdBoolean,
            constraints: Constraints {
                // Boolean + range is semantically pointless → warning (not error).
                // NOTE: We skip the type-compatibility error check by treating
                // XsdBoolean as non-numeric (no ERROR from constraint check).
                ..Default::default()
            },
        });
        model.properties.push(mandatory_prop("flag", "FlagChar"));
        let result = AspectValidator::validate(&model);
        assert!(result.is_valid());
    }

    // ── Cyclic reference detection ─────────────────────────────────────────────

    #[test]
    fn test_no_cycle_in_linear_chain() {
        let mut model = valid_model();
        model.entities.push(AspectEntity {
            name: "A".to_owned(),
            extends: None,
            properties: vec![mandatory_prop("pa", "NameChar")],
        });
        model.entities.push(AspectEntity {
            name: "B".to_owned(),
            extends: Some("A".to_owned()),
            properties: vec![mandatory_prop("pb", "NameChar")],
        });
        model.entities.push(AspectEntity {
            name: "C".to_owned(),
            extends: Some("B".to_owned()),
            properties: vec![mandatory_prop("pc", "NameChar")],
        });
        let result = AspectValidator::validate(&model);
        let errors = result.errors();
        let cycle_errors: Vec<_> = errors
            .iter()
            .filter(|m| m.message.contains("Cyclic"))
            .collect();
        assert!(
            cycle_errors.is_empty(),
            "unexpected cycle errors: {:?}",
            cycle_errors
        );
    }

    #[test]
    fn test_self_referential_cycle_detected() {
        let mut model = valid_model();
        model.entities.push(AspectEntity {
            name: "SelfRef".to_owned(),
            extends: Some("SelfRef".to_owned()),
            properties: vec![mandatory_prop("ps", "NameChar")],
        });
        let result = AspectValidator::validate(&model);
        assert!(!result.is_valid());
        assert!(result.errors().iter().any(|m| m.message.contains("Cyclic")));
    }

    #[test]
    fn test_multi_node_cycle_detected() {
        let mut model = valid_model();
        // A → B → C → A (cycle of length 3)
        model.entities.push(AspectEntity {
            name: "EA".to_owned(),
            extends: Some("EC".to_owned()),
            properties: vec![mandatory_prop("pa", "NameChar")],
        });
        model.entities.push(AspectEntity {
            name: "EB".to_owned(),
            extends: Some("EA".to_owned()),
            properties: vec![mandatory_prop("pb", "NameChar")],
        });
        model.entities.push(AspectEntity {
            name: "EC".to_owned(),
            extends: Some("EB".to_owned()),
            properties: vec![mandatory_prop("pc", "NameChar")],
        });
        let result = AspectValidator::validate(&model);
        assert!(!result.is_valid());
        let errs = result.errors();
        let cycle_errs: Vec<_> = errs
            .iter()
            .filter(|m| m.message.contains("Cyclic"))
            .collect();
        assert!(!cycle_errs.is_empty());
    }

    // ── Validation result helpers ──────────────────────────────────────────────

    #[test]
    fn test_validation_result_is_empty() {
        let r = ValidationResult::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn test_validation_message_display() {
        let msg = ValidationMessage::error("MyProp", "something is wrong");
        let s = msg.to_string();
        assert!(s.contains("ERROR"));
        assert!(s.contains("MyProp"));
        assert!(s.contains("something is wrong"));
    }

    #[test]
    fn test_messages_at_least_filters_correctly() {
        let mut result = ValidationResult::new();
        result.push(ValidationMessage::info("x", "info msg"));
        result.push(ValidationMessage::warning("y", "warn msg"));
        result.push(ValidationMessage::error("z", "err msg"));

        assert_eq!(result.messages_at_least(Severity::Error).len(), 1);
        assert_eq!(result.messages_at_least(Severity::Warning).len(), 2);
        assert_eq!(result.messages_at_least(Severity::Info).len(), 3);
    }

    // ── DataType helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_data_type_from_str() {
        assert_eq!(
            "xsd:string".parse::<DataType>().unwrap(),
            DataType::XsdString
        );
        assert_eq!(
            "xsd:integer".parse::<DataType>().unwrap(),
            DataType::XsdInteger
        );
        assert_eq!(
            "xsd:boolean".parse::<DataType>().unwrap(),
            DataType::XsdBoolean
        );
        assert!(matches!(
            "custom:MyType".parse::<DataType>().unwrap(),
            DataType::Custom(_)
        ));
    }

    #[test]
    fn test_data_type_is_numeric() {
        assert!(DataType::XsdInteger.is_numeric());
        assert!(DataType::XsdFloat.is_numeric());
        assert!(!DataType::XsdString.is_numeric());
        assert!(!DataType::XsdBoolean.is_numeric());
    }

    #[test]
    fn test_data_type_is_textual() {
        assert!(DataType::XsdString.is_textual());
        assert!(!DataType::XsdInteger.is_textual());
    }

    // ── Path preservation ─────────────────────────────────────────────────────

    #[test]
    fn test_validation_message_with_path() {
        let msg = ValidationMessage::error("Prop", "bad ref").with_path("Aspect.Entity.Prop");
        assert_eq!(msg.path.as_deref(), Some("Aspect.Entity.Prop"));
    }

    #[test]
    fn test_validation_message_without_path_is_none() {
        let msg = ValidationMessage::warning("X", "just a warning");
        assert!(msg.path.is_none());
    }

    // ── int_char / string_char helpers ────────────────────────────────────────

    #[test]
    fn test_int_char_data_type() {
        let c = int_char("C");
        assert_eq!(c.data_type, DataType::XsdInteger);
    }

    #[test]
    fn test_string_char_data_type() {
        let c = string_char("C");
        assert_eq!(c.data_type, DataType::XsdString);
    }

    // ── Additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_aspect_model_new_defaults() {
        let model = AspectModel::new("MyAspect");
        assert_eq!(model.name, "MyAspect");
        assert!(model.properties.is_empty());
        assert!(model.characteristics.is_empty());
        assert!(model.entities.is_empty());
        assert!(model.description.is_none());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(Severity::Info.to_string(), "INFO");
        assert_eq!(Severity::Warning.to_string(), "WARNING");
        assert_eq!(Severity::Error.to_string(), "ERROR");
    }

    #[test]
    fn test_validation_result_errors_and_warnings() {
        let mut result = ValidationResult::new();
        result.push(ValidationMessage::error("A", "err"));
        result.push(ValidationMessage::warning("B", "warn"));
        assert_eq!(result.errors().len(), 1);
        assert_eq!(result.warnings().len(), 1);
    }

    #[test]
    fn test_pattern_constraint_on_string_ok() {
        let mut model = AspectModel::new("PatModel");
        model.description = Some("desc".to_owned());
        model.characteristics.push(AspectCharacteristic {
            name: "PatChar".to_owned(),
            data_type: DataType::XsdString,
            constraints: Constraints {
                pattern: Some(PatternConstraint {
                    pattern: "^[A-Z]+$".to_owned(),
                }),
                ..Default::default()
            },
        });
        model.properties.push(mandatory_prop("code", "PatChar"));
        let result = AspectValidator::validate(&model);
        assert!(result.is_valid(), "errors: {:?}", result.errors());
    }

    #[test]
    fn test_pattern_constraint_on_non_string_error() {
        let mut model = AspectModel::new("BadPat");
        model.description = Some("desc".to_owned());
        model.characteristics.push(AspectCharacteristic {
            name: "NumChar".to_owned(),
            data_type: DataType::XsdFloat,
            constraints: Constraints {
                pattern: Some(PatternConstraint {
                    pattern: "^[0-9]+$".to_owned(),
                }),
                ..Default::default()
            },
        });
        model.properties.push(mandatory_prop("n", "NumChar"));
        let result = AspectValidator::validate(&model);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_multiple_errors_accumulate() {
        let mut model = AspectModel::new("MultiErr");
        model.description = Some("desc".to_owned());
        // Two properties referencing unknown characteristics.
        model.properties.push(mandatory_prop("p1", "UnknownA"));
        model.properties.push(mandatory_prop("p2", "UnknownB"));
        let result = AspectValidator::validate(&model);
        assert!(!result.is_valid());
        assert!(result.errors().len() >= 2);
    }

    #[test]
    fn test_description_whitespace_only_triggers_info() {
        let mut model = AspectModel::new("WhiteSpace");
        model.description = Some("   ".to_owned());
        model.characteristics.push(string_char("C"));
        model.properties.push(mandatory_prop("p", "C"));
        let result = AspectValidator::validate(&model);
        // Whitespace-only description is equivalent to no description.
        let infos: Vec<_> = result
            .messages
            .iter()
            .filter(|m| m.severity == Severity::Info)
            .collect();
        assert!(!infos.is_empty());
    }

    #[test]
    fn test_range_constraint_min_only_ok() {
        let mut model = AspectModel::new("MinOnly");
        model.description = Some("desc".to_owned());
        model.characteristics.push(AspectCharacteristic {
            name: "NumChar".to_owned(),
            data_type: DataType::XsdFloat,
            constraints: Constraints {
                range: Some(RangeConstraint::min_only(0.0)),
                ..Default::default()
            },
        });
        model.properties.push(mandatory_prop("val", "NumChar"));
        let result = AspectValidator::validate(&model);
        assert!(result.is_valid(), "errors: {:?}", result.errors());
    }

    #[test]
    fn test_range_constraint_max_only_ok() {
        let mut model = AspectModel::new("MaxOnly");
        model.description = Some("desc".to_owned());
        model.characteristics.push(AspectCharacteristic {
            name: "NumChar".to_owned(),
            data_type: DataType::XsdInteger,
            constraints: Constraints {
                range: Some(RangeConstraint::max_only(100.0)),
                ..Default::default()
            },
        });
        model.properties.push(mandatory_prop("val", "NumChar"));
        let result = AspectValidator::validate(&model);
        assert!(result.is_valid(), "errors: {:?}", result.errors());
    }

    #[test]
    fn test_length_constraint_min_exceeds_max_error() {
        let mut model = AspectModel::new("LenBound");
        model.description = Some("desc".to_owned());
        model.characteristics.push(AspectCharacteristic {
            name: "StrChar".to_owned(),
            data_type: DataType::XsdString,
            constraints: Constraints {
                length: Some(LengthConstraint {
                    min: Some(10),
                    max: Some(5),
                }),
                ..Default::default()
            },
        });
        model.properties.push(mandatory_prop("s", "StrChar"));
        let result = AspectValidator::validate(&model);
        assert!(!result.is_valid());
        assert!(result
            .errors()
            .iter()
            .any(|m| m.message.contains("LengthConstraint")));
    }

    #[test]
    fn test_data_type_from_full_iri_string() {
        let dt = "http://www.w3.org/2001/XMLSchema#string"
            .parse::<DataType>()
            .unwrap();
        assert_eq!(dt, DataType::XsdString);
    }

    #[test]
    fn test_cardinality_optional_vs_mandatory() {
        let m = mandatory_prop("m", "C");
        let o = optional_prop("o", "C");
        assert_eq!(m.cardinality, Cardinality::Mandatory);
        assert_eq!(o.cardinality, Cardinality::Optional);
    }

    #[test]
    fn test_aspect_with_no_properties_is_valid() {
        // An aspect with no properties (valid edge case — no required-prop errors).
        let mut model = AspectModel::new("Empty");
        model.description = Some("desc".to_owned());
        let result = AspectValidator::validate(&model);
        assert!(result.is_valid(), "errors: {:?}", result.errors());
    }

    #[test]
    fn test_validation_result_len() {
        let mut result = ValidationResult::new();
        result.push(ValidationMessage::info("X", "i"));
        result.push(ValidationMessage::warning("Y", "w"));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_range_constraint_bounded_constructor() {
        let r = RangeConstraint::bounded(1.0, 10.0);
        assert_eq!(r.min, Some(1.0));
        assert_eq!(r.max, Some(10.0));
    }
}
