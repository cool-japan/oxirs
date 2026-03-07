//! # AAS Submodel Template Library
//!
//! Provides a registry of standardized submodel templates based on the IDTA
//! (Industrial Digital Twin Association) specification library.
//!
//! Templates define the structure, semantics, and constraints for common
//! industrial submodel types, enabling:
//!
//! - Instantiation of standard-compliant submodels
//! - Validation of existing submodels against templates
//! - Discovery and search of available templates
//! - Version-aware template management
//!
//! ## Standard Templates (IDTA)
//!
//! - **Digital Nameplate** (IDTA 02006-2-0): Identification data for assets
//! - **Technical Data** (IDTA 02003-1-2): Technical specifications
//! - **Contact Information** (IDTA 02002-1-0): Organization contact details
//! - **Handover Documentation** (IDTA 02004-1-2): Document handover
//! - **Carbon Footprint** (IDTA 02023-0-9): Environmental impact
//! - **Time Series Data** (IDTA 02008-1-1): Temporal data recording
//! - **Hierarchical Structures** (IDTA 02011-1-0): Bill of materials
//! - **Software Nameplate** (IDTA 02005-1-0): Software identification

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Template types
// ---------------------------------------------------------------------------

/// A submodel template definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmodelTemplate {
    /// IDTA identifier (e.g., "IDTA 02006-2-0").
    pub idta_id: String,
    /// Semantic ID (IRI) for this template.
    pub semantic_id: String,
    /// Human-readable name.
    pub name: String,
    /// Version of the template.
    pub version: TemplateVersion,
    /// Description of the template's purpose.
    pub description: String,
    /// Required elements that must be present.
    pub required_elements: Vec<TemplateElement>,
    /// Optional elements that may be present.
    pub optional_elements: Vec<TemplateElement>,
    /// Constraints on the submodel.
    pub constraints: Vec<TemplateConstraint>,
    /// Category for searching.
    pub category: TemplateCategory,
    /// Tags for discovery.
    pub tags: Vec<String>,
}

/// Version of a template following semantic versioning.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TemplateVersion {
    /// Major version.
    pub major: u32,
    /// Minor version.
    pub minor: u32,
    /// Patch version.
    pub patch: u32,
}

impl fmt::Display for TemplateVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl TemplateVersion {
    /// Create a new version.
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Check if this version is compatible with another (same major).
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major
    }
}

/// An element definition within a template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateElement {
    /// Short identifier.
    pub id_short: String,
    /// Semantic ID for the element.
    pub semantic_id: Option<String>,
    /// Element type.
    pub element_type: ElementType,
    /// Value type (for properties).
    pub value_type: Option<ValueType>,
    /// Description.
    pub description: String,
    /// Multiplicity constraint.
    pub multiplicity: Multiplicity,
    /// Nested elements (for collections).
    pub children: Vec<TemplateElement>,
    /// Example value.
    pub example_value: Option<String>,
}

/// Type of a submodel element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElementType {
    /// A simple property with a value.
    Property,
    /// A collection of elements.
    Collection,
    /// An operation with input/output.
    Operation,
    /// A reference to another element.
    ReferenceElement,
    /// A file reference.
    File,
    /// A blob (binary large object).
    Blob,
    /// A multi-language property.
    MultiLanguageProperty,
    /// A range of values.
    Range,
    /// An entity with properties.
    Entity,
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Property => write!(f, "Property"),
            Self::Collection => write!(f, "SubmodelElementCollection"),
            Self::Operation => write!(f, "Operation"),
            Self::ReferenceElement => write!(f, "ReferenceElement"),
            Self::File => write!(f, "File"),
            Self::Blob => write!(f, "Blob"),
            Self::MultiLanguageProperty => write!(f, "MultiLanguageProperty"),
            Self::Range => write!(f, "Range"),
            Self::Entity => write!(f, "Entity"),
        }
    }
}

/// Value type for properties.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValueType {
    /// xs:string
    String,
    /// xs:integer
    Integer,
    /// xs:double
    Double,
    /// xs:boolean
    Boolean,
    /// xs:dateTime
    DateTime,
    /// xs:date
    Date,
    /// xs:anyURI
    AnyUri,
    /// Custom datatype IRI.
    Custom(String),
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String => write!(f, "xs:string"),
            Self::Integer => write!(f, "xs:integer"),
            Self::Double => write!(f, "xs:double"),
            Self::Boolean => write!(f, "xs:boolean"),
            Self::DateTime => write!(f, "xs:dateTime"),
            Self::Date => write!(f, "xs:date"),
            Self::AnyUri => write!(f, "xs:anyURI"),
            Self::Custom(dt) => write!(f, "{dt}"),
        }
    }
}

/// Multiplicity constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Multiplicity {
    /// Exactly one (mandatory).
    One,
    /// Zero or one (optional).
    ZeroOrOne,
    /// Zero or more.
    ZeroOrMore,
    /// One or more.
    OneOrMore,
}

impl fmt::Display for Multiplicity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::One => write!(f, "[1]"),
            Self::ZeroOrOne => write!(f, "[0..1]"),
            Self::ZeroOrMore => write!(f, "[0..*]"),
            Self::OneOrMore => write!(f, "[1..*]"),
        }
    }
}

impl Multiplicity {
    /// Check if this multiplicity requires at least one element.
    pub fn is_required(&self) -> bool {
        matches!(self, Self::One | Self::OneOrMore)
    }
}

/// Constraints on template values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemplateConstraint {
    /// Element must have a specific value.
    FixedValue {
        /// The target element identifier to constrain.
        element: String,
        /// The fixed value the element must have.
        value: String,
    },
    /// Element value must match a regex pattern.
    Pattern {
        /// The target element identifier to constrain.
        element: String,
        /// The regex pattern the element value must match.
        pattern: String,
    },
    /// Element value must be from an enumeration.
    Enumeration {
        /// The target element identifier to constrain.
        element: String,
        /// The set of allowed values for the element.
        allowed_values: Vec<String>,
    },
    /// Element value must be within a numeric range.
    NumericRange {
        /// The target element identifier to constrain.
        element: String,
        /// The minimum allowed value (inclusive), or `None` for unbounded.
        min: Option<f64>,
        /// The maximum allowed value (inclusive), or `None` for unbounded.
        max: Option<f64>,
    },
    /// Element string length constraint.
    StringLength {
        /// The target element identifier to constrain.
        element: String,
        /// The minimum allowed string length, or `None` for unbounded.
        min_length: Option<usize>,
        /// The maximum allowed string length, or `None` for unbounded.
        max_length: Option<usize>,
    },
    /// Conditional requirement: if element A is present, B must be present.
    ConditionalRequired {
        /// The element whose presence triggers the requirement.
        condition_element: String,
        /// The element that must be present when the condition element exists.
        required_element: String,
    },
}

/// Category for template classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemplateCategory {
    /// Identification and nameplate data.
    Identification,
    /// Technical specifications.
    TechnicalData,
    /// Documentation and manuals.
    Documentation,
    /// Environmental and sustainability data.
    Sustainability,
    /// Time series and sensor data.
    TimeSeries,
    /// Structural information (BOM, hierarchy).
    Structure,
    /// Contact and organizational data.
    ContactInfo,
    /// Software and firmware.
    Software,
    /// Safety and compliance.
    SafetyCompliance,
    /// Other or custom.
    Other,
}

impl fmt::Display for TemplateCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identification => write!(f, "Identification"),
            Self::TechnicalData => write!(f, "Technical Data"),
            Self::Documentation => write!(f, "Documentation"),
            Self::Sustainability => write!(f, "Sustainability"),
            Self::TimeSeries => write!(f, "Time Series"),
            Self::Structure => write!(f, "Structure"),
            Self::ContactInfo => write!(f, "Contact Information"),
            Self::Software => write!(f, "Software"),
            Self::SafetyCompliance => write!(f, "Safety & Compliance"),
            Self::Other => write!(f, "Other"),
        }
    }
}

// ---------------------------------------------------------------------------
// Validation result
// ---------------------------------------------------------------------------

/// Result of validating a submodel instance against a template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateValidationResult {
    /// Whether the submodel conforms to the template.
    pub is_valid: bool,
    /// Template that was validated against.
    pub template_id: String,
    /// Template version.
    pub template_version: TemplateVersion,
    /// Conformance score (0.0–1.0).
    pub conformance_score: f64,
    /// Errors that prevent conformance.
    pub errors: Vec<TemplateValidationError>,
    /// Warnings (conformant but potentially problematic).
    pub warnings: Vec<String>,
    /// Elements that matched the template.
    pub matched_elements: usize,
    /// Required elements that were missing.
    pub missing_required: Vec<String>,
    /// Extra elements not in the template.
    pub extra_elements: Vec<String>,
}

/// A validation error against a template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateValidationError {
    /// Element path where the error occurred.
    pub element_path: String,
    /// Error message.
    pub message: String,
    /// Error severity.
    pub severity: ValidationSeverity,
}

/// Severity levels for template validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Critical: prevents conformance.
    Error,
    /// Warning: conformant but not ideal.
    Warning,
    /// Informational.
    Info,
}

// ---------------------------------------------------------------------------
// Template Registry
// ---------------------------------------------------------------------------

/// A registry of submodel templates with search and version management.
pub struct TemplateRegistry {
    templates: HashMap<String, Vec<SubmodelTemplate>>,
    /// Aliases for template lookup.
    aliases: HashMap<String, String>,
}

impl TemplateRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            aliases: HashMap::new(),
        }
    }

    /// Create a registry pre-populated with IDTA standard templates.
    pub fn with_standards() -> Self {
        let mut registry = Self::new();
        registry.register(Self::digital_nameplate_template());
        registry.register(Self::technical_data_template());
        registry.register(Self::contact_information_template());
        registry.register(Self::carbon_footprint_template());
        registry.register(Self::time_series_data_template());
        registry.register(Self::hierarchical_structures_template());
        registry.register(Self::software_nameplate_template());
        registry.register(Self::handover_documentation_template());
        registry
    }

    /// Register a template. Multiple versions of the same template
    /// (by semantic_id) are kept.
    pub fn register(&mut self, template: SubmodelTemplate) {
        // Register by semantic ID
        self.templates
            .entry(template.semantic_id.clone())
            .or_default()
            .push(template.clone());

        // Register alias by IDTA ID
        self.aliases
            .insert(template.idta_id.clone(), template.semantic_id.clone());

        // Register alias by name (lowercase)
        self.aliases
            .insert(template.name.to_lowercase(), template.semantic_id.clone());
    }

    /// Get the latest version of a template by semantic ID or alias.
    pub fn get(&self, id: &str) -> Option<&SubmodelTemplate> {
        let semantic_id = self.aliases.get(id).map(|s| s.as_str()).unwrap_or(id);
        self.templates
            .get(semantic_id)
            .and_then(|versions| versions.iter().max_by_key(|t| &t.version))
    }

    /// Get a specific version of a template.
    pub fn get_version(&self, id: &str, version: &TemplateVersion) -> Option<&SubmodelTemplate> {
        let semantic_id = self.aliases.get(id).map(|s| s.as_str()).unwrap_or(id);
        self.templates
            .get(semantic_id)
            .and_then(|versions| versions.iter().find(|t| &t.version == version))
    }

    /// List all available templates (latest versions only).
    pub fn list_all(&self) -> Vec<&SubmodelTemplate> {
        self.templates
            .values()
            .filter_map(|versions| versions.iter().max_by_key(|t| &t.version))
            .collect()
    }

    /// Search templates by keyword (matches name, description, tags).
    pub fn search(&self, query: &str) -> Vec<&SubmodelTemplate> {
        let query_lower = query.to_lowercase();
        self.list_all()
            .into_iter()
            .filter(|t| {
                t.name.to_lowercase().contains(&query_lower)
                    || t.description.to_lowercase().contains(&query_lower)
                    || t.tags
                        .iter()
                        .any(|tag| tag.to_lowercase().contains(&query_lower))
                    || t.idta_id.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// Search templates by category.
    pub fn by_category(&self, category: TemplateCategory) -> Vec<&SubmodelTemplate> {
        self.list_all()
            .into_iter()
            .filter(|t| t.category == category)
            .collect()
    }

    /// Total number of unique templates (not counting versions).
    pub fn count(&self) -> usize {
        self.templates.len()
    }

    /// Total number of template versions across all templates.
    pub fn version_count(&self) -> usize {
        self.templates.values().map(|v| v.len()).sum()
    }

    /// Validate a submodel instance (represented as key-value pairs)
    /// against a template.
    pub fn validate_instance(
        &self,
        template_id: &str,
        elements: &HashMap<String, String>,
    ) -> Option<TemplateValidationResult> {
        let template = self.get(template_id)?;

        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut matched = 0usize;
        let mut missing_required = Vec::new();

        // Check required elements
        for req in &template.required_elements {
            if elements.contains_key(&req.id_short) {
                matched += 1;
            } else {
                missing_required.push(req.id_short.clone());
                errors.push(TemplateValidationError {
                    element_path: req.id_short.clone(),
                    message: format!(
                        "Required element '{}' is missing (type: {})",
                        req.id_short, req.element_type
                    ),
                    severity: ValidationSeverity::Error,
                });
            }
        }

        // Check optional elements presence
        for opt in &template.optional_elements {
            if elements.contains_key(&opt.id_short) {
                matched += 1;
            }
        }

        // Check for extra elements
        let template_ids: std::collections::HashSet<_> = template
            .required_elements
            .iter()
            .chain(template.optional_elements.iter())
            .map(|e| e.id_short.as_str())
            .collect();

        let extra: Vec<_> = elements
            .keys()
            .filter(|k| !template_ids.contains(k.as_str()))
            .cloned()
            .collect();

        if !extra.is_empty() {
            warnings.push(format!(
                "Found {} elements not in template: {}",
                extra.len(),
                extra.join(", ")
            ));
        }

        // Check constraints
        for constraint in &template.constraints {
            if let Some(err) = self.check_constraint(constraint, elements) {
                errors.push(err);
            }
        }

        let total_template_elements =
            template.required_elements.len() + template.optional_elements.len();
        let conformance_score = if total_template_elements == 0 {
            1.0
        } else {
            matched as f64 / total_template_elements as f64
        };

        Some(TemplateValidationResult {
            is_valid: errors.is_empty(),
            template_id: template.semantic_id.clone(),
            template_version: template.version.clone(),
            conformance_score,
            errors,
            warnings,
            matched_elements: matched,
            missing_required,
            extra_elements: extra,
        })
    }

    fn check_constraint(
        &self,
        constraint: &TemplateConstraint,
        elements: &HashMap<String, String>,
    ) -> Option<TemplateValidationError> {
        match constraint {
            TemplateConstraint::FixedValue { element, value } => {
                if let Some(actual) = elements.get(element) {
                    if actual != value {
                        return Some(TemplateValidationError {
                            element_path: element.clone(),
                            message: format!("Expected fixed value '{value}', got '{actual}'"),
                            severity: ValidationSeverity::Error,
                        });
                    }
                }
                None
            }
            TemplateConstraint::Enumeration {
                element,
                allowed_values,
            } => {
                if let Some(actual) = elements.get(element) {
                    if !allowed_values.contains(actual) {
                        return Some(TemplateValidationError {
                            element_path: element.clone(),
                            message: format!(
                                "Value '{actual}' not in allowed values: {:?}",
                                allowed_values
                            ),
                            severity: ValidationSeverity::Error,
                        });
                    }
                }
                None
            }
            TemplateConstraint::NumericRange { element, min, max } => {
                if let Some(actual) = elements.get(element) {
                    if let Ok(val) = actual.parse::<f64>() {
                        if let Some(mn) = min {
                            if val < *mn {
                                return Some(TemplateValidationError {
                                    element_path: element.clone(),
                                    message: format!("Value {val} is below minimum {mn}"),
                                    severity: ValidationSeverity::Error,
                                });
                            }
                        }
                        if let Some(mx) = max {
                            if val > *mx {
                                return Some(TemplateValidationError {
                                    element_path: element.clone(),
                                    message: format!("Value {val} exceeds maximum {mx}"),
                                    severity: ValidationSeverity::Error,
                                });
                            }
                        }
                    }
                }
                None
            }
            TemplateConstraint::StringLength {
                element,
                min_length,
                max_length,
            } => {
                if let Some(actual) = elements.get(element) {
                    if let Some(mn) = min_length {
                        if actual.len() < *mn {
                            return Some(TemplateValidationError {
                                element_path: element.clone(),
                                message: format!(
                                    "String length {} is below minimum {mn}",
                                    actual.len()
                                ),
                                severity: ValidationSeverity::Error,
                            });
                        }
                    }
                    if let Some(mx) = max_length {
                        if actual.len() > *mx {
                            return Some(TemplateValidationError {
                                element_path: element.clone(),
                                message: format!(
                                    "String length {} exceeds maximum {mx}",
                                    actual.len()
                                ),
                                severity: ValidationSeverity::Error,
                            });
                        }
                    }
                }
                None
            }
            TemplateConstraint::Pattern { element, pattern } => {
                if let Some(actual) = elements.get(element) {
                    if let Ok(re) = regex::Regex::new(pattern) {
                        if !re.is_match(actual) {
                            return Some(TemplateValidationError {
                                element_path: element.clone(),
                                message: format!(
                                    "Value '{actual}' does not match pattern '{pattern}'"
                                ),
                                severity: ValidationSeverity::Error,
                            });
                        }
                    }
                }
                None
            }
            TemplateConstraint::ConditionalRequired {
                condition_element,
                required_element,
            } => {
                if elements.contains_key(condition_element)
                    && !elements.contains_key(required_element)
                {
                    return Some(TemplateValidationError {
                        element_path: required_element.clone(),
                        message: format!(
                            "Element '{required_element}' is required when '{condition_element}' is present"
                        ),
                        severity: ValidationSeverity::Error,
                    });
                }
                None
            }
        }
    }

    // ── Standard template definitions ─────────────────────────────────────

    /// IDTA 02006-2-0: Digital Nameplate
    fn digital_nameplate_template() -> SubmodelTemplate {
        SubmodelTemplate {
            idta_id: "IDTA 02006-2-0".to_string(),
            semantic_id: "https://admin-shell.io/zvei/nameplate/2/0/Nameplate".to_string(),
            name: "Digital Nameplate".to_string(),
            version: TemplateVersion::new(2, 0, 0),
            description: "Digital nameplate data for asset identification per IEC 61406"
                .to_string(),
            required_elements: vec![
                TemplateElement {
                    id_short: "ManufacturerName".to_string(),
                    semantic_id: Some("0173-1#02-AAO677#002".to_string()),
                    element_type: ElementType::MultiLanguageProperty,
                    value_type: Some(ValueType::String),
                    description: "Legally valid manufacturer name".to_string(),
                    multiplicity: Multiplicity::One,
                    children: vec![],
                    example_value: Some("Siemens AG".to_string()),
                },
                TemplateElement {
                    id_short: "ManufacturerProductDesignation".to_string(),
                    semantic_id: Some("0173-1#02-AAW338#001".to_string()),
                    element_type: ElementType::MultiLanguageProperty,
                    value_type: Some(ValueType::String),
                    description: "Short product name given by the manufacturer".to_string(),
                    multiplicity: Multiplicity::One,
                    children: vec![],
                    example_value: Some("SIMATIC S7-1500".to_string()),
                },
                TemplateElement {
                    id_short: "SerialNumber".to_string(),
                    semantic_id: Some("0173-1#02-AAM556#002".to_string()),
                    element_type: ElementType::Property,
                    value_type: Some(ValueType::String),
                    description: "Unique serial number".to_string(),
                    multiplicity: Multiplicity::One,
                    children: vec![],
                    example_value: Some("SN-20240101-001".to_string()),
                },
            ],
            optional_elements: vec![
                TemplateElement {
                    id_short: "YearOfConstruction".to_string(),
                    semantic_id: Some("0173-1#02-AAP906#001".to_string()),
                    element_type: ElementType::Property,
                    value_type: Some(ValueType::String),
                    description: "Year of construction".to_string(),
                    multiplicity: Multiplicity::ZeroOrOne,
                    children: vec![],
                    example_value: Some("2024".to_string()),
                },
                TemplateElement {
                    id_short: "CompanyLogo".to_string(),
                    semantic_id: None,
                    element_type: ElementType::File,
                    value_type: None,
                    description: "Company logo image".to_string(),
                    multiplicity: Multiplicity::ZeroOrOne,
                    children: vec![],
                    example_value: None,
                },
            ],
            constraints: vec![],
            category: TemplateCategory::Identification,
            tags: vec![
                "nameplate".to_string(),
                "identification".to_string(),
                "IEC 61406".to_string(),
            ],
        }
    }

    /// IDTA 02003-1-2: Technical Data
    fn technical_data_template() -> SubmodelTemplate {
        SubmodelTemplate {
            idta_id: "IDTA 02003-1-2".to_string(),
            semantic_id: "https://admin-shell.io/ZVEI/TechnicalData/Submodel/1/2".to_string(),
            name: "Technical Data".to_string(),
            version: TemplateVersion::new(1, 2, 0),
            description: "Technical data submodel for product specifications".to_string(),
            required_elements: vec![
                TemplateElement {
                    id_short: "GeneralInformation".to_string(),
                    semantic_id: None,
                    element_type: ElementType::Collection,
                    value_type: None,
                    description: "General information about the product".to_string(),
                    multiplicity: Multiplicity::One,
                    children: vec![TemplateElement {
                        id_short: "ManufacturerName".to_string(),
                        semantic_id: None,
                        element_type: ElementType::Property,
                        value_type: Some(ValueType::String),
                        description: "Manufacturer name".to_string(),
                        multiplicity: Multiplicity::One,
                        children: vec![],
                        example_value: None,
                    }],
                    example_value: None,
                },
                TemplateElement {
                    id_short: "TechnicalProperties".to_string(),
                    semantic_id: None,
                    element_type: ElementType::Collection,
                    value_type: None,
                    description: "Technical characteristics and properties".to_string(),
                    multiplicity: Multiplicity::One,
                    children: vec![],
                    example_value: None,
                },
            ],
            optional_elements: vec![TemplateElement {
                id_short: "FurtherInformation".to_string(),
                semantic_id: None,
                element_type: ElementType::Collection,
                value_type: None,
                description: "Additional informational text".to_string(),
                multiplicity: Multiplicity::ZeroOrOne,
                children: vec![],
                example_value: None,
            }],
            constraints: vec![],
            category: TemplateCategory::TechnicalData,
            tags: vec!["technical".to_string(), "specifications".to_string()],
        }
    }

    /// IDTA 02002-1-0: Contact Information
    fn contact_information_template() -> SubmodelTemplate {
        SubmodelTemplate {
            idta_id: "IDTA 02002-1-0".to_string(),
            semantic_id: "https://admin-shell.io/zvei/nameplate/1/0/ContactInformations"
                .to_string(),
            name: "Contact Information".to_string(),
            version: TemplateVersion::new(1, 0, 0),
            description: "Contact information for organizations and persons".to_string(),
            required_elements: vec![TemplateElement {
                id_short: "ContactInformation".to_string(),
                semantic_id: None,
                element_type: ElementType::Collection,
                value_type: None,
                description: "Contact information collection".to_string(),
                multiplicity: Multiplicity::OneOrMore,
                children: vec![],
                example_value: None,
            }],
            optional_elements: vec![
                TemplateElement {
                    id_short: "Phone".to_string(),
                    semantic_id: None,
                    element_type: ElementType::Collection,
                    value_type: None,
                    description: "Phone contact".to_string(),
                    multiplicity: Multiplicity::ZeroOrMore,
                    children: vec![],
                    example_value: None,
                },
                TemplateElement {
                    id_short: "Email".to_string(),
                    semantic_id: None,
                    element_type: ElementType::Property,
                    value_type: Some(ValueType::String),
                    description: "Email address".to_string(),
                    multiplicity: Multiplicity::ZeroOrMore,
                    children: vec![],
                    example_value: Some("info@example.com".to_string()),
                },
            ],
            constraints: vec![],
            category: TemplateCategory::ContactInfo,
            tags: vec!["contact".to_string(), "organization".to_string()],
        }
    }

    /// IDTA 02023-0-9: Carbon Footprint
    fn carbon_footprint_template() -> SubmodelTemplate {
        SubmodelTemplate {
            idta_id: "IDTA 02023-0-9".to_string(),
            semantic_id: "https://admin-shell.io/idta/CarbonFootprint/CarbonFootprint/0/9"
                .to_string(),
            name: "Carbon Footprint".to_string(),
            version: TemplateVersion::new(0, 9, 0),
            description: "Carbon footprint data for sustainability reporting".to_string(),
            required_elements: vec![TemplateElement {
                id_short: "ProductCarbonFootprint".to_string(),
                semantic_id: None,
                element_type: ElementType::Collection,
                value_type: None,
                description: "Product carbon footprint data".to_string(),
                multiplicity: Multiplicity::One,
                children: vec![],
                example_value: None,
            }],
            optional_elements: vec![TemplateElement {
                id_short: "TransportCarbonFootprint".to_string(),
                semantic_id: None,
                element_type: ElementType::Collection,
                value_type: None,
                description: "Transport-related carbon footprint".to_string(),
                multiplicity: Multiplicity::ZeroOrOne,
                children: vec![],
                example_value: None,
            }],
            constraints: vec![TemplateConstraint::NumericRange {
                element: "CO2Equivalent".to_string(),
                min: Some(0.0),
                max: None,
            }],
            category: TemplateCategory::Sustainability,
            tags: vec![
                "carbon".to_string(),
                "sustainability".to_string(),
                "environment".to_string(),
                "CO2".to_string(),
            ],
        }
    }

    /// IDTA 02008-1-1: Time Series Data
    fn time_series_data_template() -> SubmodelTemplate {
        SubmodelTemplate {
            idta_id: "IDTA 02008-1-1".to_string(),
            semantic_id: "https://admin-shell.io/idta/TimeSeries/1/1".to_string(),
            name: "Time Series Data".to_string(),
            version: TemplateVersion::new(1, 1, 0),
            description: "Time series data submodel for temporal measurements".to_string(),
            required_elements: vec![TemplateElement {
                id_short: "TimeSeries".to_string(),
                semantic_id: None,
                element_type: ElementType::Collection,
                value_type: None,
                description: "Time series configuration and data".to_string(),
                multiplicity: Multiplicity::One,
                children: vec![],
                example_value: None,
            }],
            optional_elements: vec![TemplateElement {
                id_short: "Metadata".to_string(),
                semantic_id: None,
                element_type: ElementType::Collection,
                value_type: None,
                description: "Time series metadata".to_string(),
                multiplicity: Multiplicity::ZeroOrOne,
                children: vec![],
                example_value: None,
            }],
            constraints: vec![],
            category: TemplateCategory::TimeSeries,
            tags: vec![
                "timeseries".to_string(),
                "sensor".to_string(),
                "measurement".to_string(),
            ],
        }
    }

    /// IDTA 02011-1-0: Hierarchical Structures
    fn hierarchical_structures_template() -> SubmodelTemplate {
        SubmodelTemplate {
            idta_id: "IDTA 02011-1-0".to_string(),
            semantic_id: "https://admin-shell.io/idta/HierarchicalStructures/1/0/Submodel"
                .to_string(),
            name: "Hierarchical Structures".to_string(),
            version: TemplateVersion::new(1, 0, 0),
            description: "Bill of materials and hierarchical asset structures".to_string(),
            required_elements: vec![TemplateElement {
                id_short: "EntryNode".to_string(),
                semantic_id: None,
                element_type: ElementType::Entity,
                value_type: None,
                description: "Root entry node of the hierarchy".to_string(),
                multiplicity: Multiplicity::One,
                children: vec![],
                example_value: None,
            }],
            optional_elements: vec![TemplateElement {
                id_short: "ArcheType".to_string(),
                semantic_id: None,
                element_type: ElementType::Property,
                value_type: Some(ValueType::String),
                description: "Type of hierarchical structure".to_string(),
                multiplicity: Multiplicity::ZeroOrOne,
                children: vec![],
                example_value: Some("FullBoM".to_string()),
            }],
            constraints: vec![TemplateConstraint::Enumeration {
                element: "ArcheType".to_string(),
                allowed_values: vec![
                    "FullBoM".to_string(),
                    "OneDown".to_string(),
                    "OneUp".to_string(),
                ],
            }],
            category: TemplateCategory::Structure,
            tags: vec![
                "hierarchy".to_string(),
                "bom".to_string(),
                "structure".to_string(),
            ],
        }
    }

    /// IDTA 02005-1-0: Software Nameplate
    fn software_nameplate_template() -> SubmodelTemplate {
        SubmodelTemplate {
            idta_id: "IDTA 02005-1-0".to_string(),
            semantic_id: "https://admin-shell.io/idta/SoftwareNameplate/1/0".to_string(),
            name: "Software Nameplate".to_string(),
            version: TemplateVersion::new(1, 0, 0),
            description: "Software identification and version information".to_string(),
            required_elements: vec![
                TemplateElement {
                    id_short: "SoftwareName".to_string(),
                    semantic_id: None,
                    element_type: ElementType::Property,
                    value_type: Some(ValueType::String),
                    description: "Name of the software".to_string(),
                    multiplicity: Multiplicity::One,
                    children: vec![],
                    example_value: Some("FirmwareX".to_string()),
                },
                TemplateElement {
                    id_short: "SoftwareVersion".to_string(),
                    semantic_id: None,
                    element_type: ElementType::Property,
                    value_type: Some(ValueType::String),
                    description: "Version of the software".to_string(),
                    multiplicity: Multiplicity::One,
                    children: vec![],
                    example_value: Some("2.1.0".to_string()),
                },
            ],
            optional_elements: vec![TemplateElement {
                id_short: "SoftwareType".to_string(),
                semantic_id: None,
                element_type: ElementType::Property,
                value_type: Some(ValueType::String),
                description: "Type of software".to_string(),
                multiplicity: Multiplicity::ZeroOrOne,
                children: vec![],
                example_value: Some("Firmware".to_string()),
            }],
            constraints: vec![TemplateConstraint::StringLength {
                element: "SoftwareVersion".to_string(),
                min_length: Some(1),
                max_length: Some(100),
            }],
            category: TemplateCategory::Software,
            tags: vec![
                "software".to_string(),
                "firmware".to_string(),
                "version".to_string(),
            ],
        }
    }

    /// IDTA 02004-1-2: Handover Documentation
    fn handover_documentation_template() -> SubmodelTemplate {
        SubmodelTemplate {
            idta_id: "IDTA 02004-1-2".to_string(),
            semantic_id: "https://admin-shell.io/zvei/nameplate/1/0/HandoverDocumentation"
                .to_string(),
            name: "Handover Documentation".to_string(),
            version: TemplateVersion::new(1, 2, 0),
            description: "Documentation for asset handover".to_string(),
            required_elements: vec![TemplateElement {
                id_short: "Document".to_string(),
                semantic_id: None,
                element_type: ElementType::Collection,
                value_type: None,
                description: "Document entry".to_string(),
                multiplicity: Multiplicity::OneOrMore,
                children: vec![],
                example_value: None,
            }],
            optional_elements: vec![],
            constraints: vec![],
            category: TemplateCategory::Documentation,
            tags: vec![
                "documentation".to_string(),
                "handover".to_string(),
                "manual".to_string(),
            ],
        }
    }
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn std_registry() -> TemplateRegistry {
        TemplateRegistry::with_standards()
    }

    // ── Registry basic tests ──────────────────────────────────────────────

    #[test]
    fn test_registry_with_standards_count() {
        let reg = std_registry();
        assert_eq!(reg.count(), 8);
    }

    #[test]
    fn test_registry_new_empty() {
        let reg = TemplateRegistry::new();
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_registry_register_custom() {
        let mut reg = TemplateRegistry::new();
        reg.register(SubmodelTemplate {
            idta_id: "CUSTOM-001".to_string(),
            semantic_id: "https://example.org/CustomTemplate".to_string(),
            name: "Custom Template".to_string(),
            version: TemplateVersion::new(1, 0, 0),
            description: "A custom template".to_string(),
            required_elements: vec![],
            optional_elements: vec![],
            constraints: vec![],
            category: TemplateCategory::Other,
            tags: vec!["custom".to_string()],
        });
        assert_eq!(reg.count(), 1);
    }

    // ── Get by ID ─────────────────────────────────────────────────────────

    #[test]
    fn test_get_by_semantic_id() {
        let reg = std_registry();
        let tmpl = reg
            .get("https://admin-shell.io/zvei/nameplate/2/0/Nameplate")
            .expect("should find nameplate");
        assert_eq!(tmpl.name, "Digital Nameplate");
    }

    #[test]
    fn test_get_by_idta_id() {
        let reg = std_registry();
        let tmpl = reg.get("IDTA 02006-2-0").expect("should find by IDTA ID");
        assert_eq!(tmpl.name, "Digital Nameplate");
    }

    #[test]
    fn test_get_by_name_alias() {
        let reg = std_registry();
        let tmpl = reg.get("digital nameplate").expect("should find by name");
        assert_eq!(tmpl.idta_id, "IDTA 02006-2-0");
    }

    #[test]
    fn test_get_not_found() {
        let reg = std_registry();
        assert!(reg.get("nonexistent").is_none());
    }

    // ── Version management ────────────────────────────────────────────────

    #[test]
    fn test_get_specific_version() {
        let mut reg = TemplateRegistry::new();
        reg.register(SubmodelTemplate {
            idta_id: "T-1".to_string(),
            semantic_id: "https://ex.org/t".to_string(),
            name: "Test".to_string(),
            version: TemplateVersion::new(1, 0, 0),
            description: "v1".to_string(),
            required_elements: vec![],
            optional_elements: vec![],
            constraints: vec![],
            category: TemplateCategory::Other,
            tags: vec![],
        });
        reg.register(SubmodelTemplate {
            idta_id: "T-1".to_string(),
            semantic_id: "https://ex.org/t".to_string(),
            name: "Test".to_string(),
            version: TemplateVersion::new(2, 0, 0),
            description: "v2".to_string(),
            required_elements: vec![],
            optional_elements: vec![],
            constraints: vec![],
            category: TemplateCategory::Other,
            tags: vec![],
        });

        let v1 = reg
            .get_version("https://ex.org/t", &TemplateVersion::new(1, 0, 0))
            .expect("v1");
        assert_eq!(v1.description, "v1");

        let latest = reg.get("https://ex.org/t").expect("latest");
        assert_eq!(latest.description, "v2");
    }

    #[test]
    fn test_version_count() {
        let mut reg = TemplateRegistry::new();
        for i in 0..3 {
            reg.register(SubmodelTemplate {
                idta_id: "T".to_string(),
                semantic_id: "https://ex.org/t".to_string(),
                name: "Test".to_string(),
                version: TemplateVersion::new(i, 0, 0),
                description: format!("v{i}"),
                required_elements: vec![],
                optional_elements: vec![],
                constraints: vec![],
                category: TemplateCategory::Other,
                tags: vec![],
            });
        }
        assert_eq!(reg.version_count(), 3);
        assert_eq!(reg.count(), 1); // One unique semantic ID
    }

    // ── Search ────────────────────────────────────────────────────────────

    #[test]
    fn test_search_by_keyword() {
        let reg = std_registry();
        let results = reg.search("nameplate");
        assert!(results.len() >= 2); // Digital Nameplate + Software Nameplate
    }

    #[test]
    fn test_search_by_tag() {
        let reg = std_registry();
        let results = reg.search("sustainability");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Carbon Footprint");
    }

    #[test]
    fn test_search_by_idta_id() {
        let reg = std_registry();
        let results = reg.search("02008");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Time Series Data");
    }

    #[test]
    fn test_search_no_results() {
        let reg = std_registry();
        let results = reg.search("zzz_nonexistent_zzz");
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_case_insensitive() {
        let reg = std_registry();
        let results = reg.search("CARBON");
        assert_eq!(results.len(), 1);
    }

    // ── By category ───────────────────────────────────────────────────────

    #[test]
    fn test_by_category_identification() {
        let reg = std_registry();
        let results = reg.by_category(TemplateCategory::Identification);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Digital Nameplate");
    }

    #[test]
    fn test_by_category_sustainability() {
        let reg = std_registry();
        let results = reg.by_category(TemplateCategory::Sustainability);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_by_category_empty() {
        let reg = std_registry();
        let results = reg.by_category(TemplateCategory::SafetyCompliance);
        assert!(results.is_empty());
    }

    // ── Validation ────────────────────────────────────────────────────────

    #[test]
    fn test_validate_valid_instance() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("ManufacturerName".to_string(), "Siemens AG".to_string());
        elements.insert(
            "ManufacturerProductDesignation".to_string(),
            "S7-1500".to_string(),
        );
        elements.insert("SerialNumber".to_string(), "SN-001".to_string());

        let result = reg
            .validate_instance("IDTA 02006-2-0", &elements)
            .expect("template found");
        assert!(result.is_valid);
        assert_eq!(result.matched_elements, 3);
        assert!(result.missing_required.is_empty());
    }

    #[test]
    fn test_validate_missing_required() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("ManufacturerName".to_string(), "Siemens AG".to_string());
        // Missing ManufacturerProductDesignation and SerialNumber

        let result = reg
            .validate_instance("IDTA 02006-2-0", &elements)
            .expect("template found");
        assert!(!result.is_valid);
        assert_eq!(result.missing_required.len(), 2);
    }

    #[test]
    fn test_validate_with_optional_elements() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("ManufacturerName".to_string(), "Test".to_string());
        elements.insert(
            "ManufacturerProductDesignation".to_string(),
            "Prod".to_string(),
        );
        elements.insert("SerialNumber".to_string(), "SN-1".to_string());
        elements.insert("YearOfConstruction".to_string(), "2024".to_string());

        let result = reg
            .validate_instance("IDTA 02006-2-0", &elements)
            .expect("template found");
        assert!(result.is_valid);
        assert_eq!(result.matched_elements, 4);
    }

    #[test]
    fn test_validate_extra_elements_warning() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("ManufacturerName".to_string(), "Test".to_string());
        elements.insert(
            "ManufacturerProductDesignation".to_string(),
            "Prod".to_string(),
        );
        elements.insert("SerialNumber".to_string(), "SN-1".to_string());
        elements.insert("UnknownElement".to_string(), "extra".to_string());

        let result = reg
            .validate_instance("IDTA 02006-2-0", &elements)
            .expect("template found");
        assert!(result.is_valid);
        assert!(!result.warnings.is_empty());
        assert_eq!(result.extra_elements.len(), 1);
    }

    #[test]
    fn test_validate_enum_constraint_valid() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("EntryNode".to_string(), "root".to_string());
        elements.insert("ArcheType".to_string(), "FullBoM".to_string());

        let result = reg
            .validate_instance("IDTA 02011-1-0", &elements)
            .expect("found");
        assert!(result.is_valid);
    }

    #[test]
    fn test_validate_enum_constraint_invalid() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("EntryNode".to_string(), "root".to_string());
        elements.insert("ArcheType".to_string(), "InvalidType".to_string());

        let result = reg
            .validate_instance("IDTA 02011-1-0", &elements)
            .expect("found");
        assert!(!result.is_valid);
    }

    #[test]
    fn test_validate_numeric_range() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("ProductCarbonFootprint".to_string(), "data".to_string());
        elements.insert("CO2Equivalent".to_string(), "-5".to_string());

        let result = reg
            .validate_instance("IDTA 02023-0-9", &elements)
            .expect("found");
        assert!(!result.is_valid);
    }

    #[test]
    fn test_validate_string_length() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("SoftwareName".to_string(), "MyApp".to_string());
        elements.insert("SoftwareVersion".to_string(), String::new()); // empty!

        let result = reg
            .validate_instance("IDTA 02005-1-0", &elements)
            .expect("found");
        assert!(!result.is_valid);
    }

    #[test]
    fn test_validate_nonexistent_template() {
        let reg = std_registry();
        let elements = HashMap::new();
        assert!(reg.validate_instance("nonexistent", &elements).is_none());
    }

    #[test]
    fn test_conformance_score() {
        let reg = std_registry();
        let mut elements = HashMap::new();
        elements.insert("ManufacturerName".to_string(), "Test".to_string());
        // Missing 2 of 3 required + 2 optional = 5 total, 1 matched

        let result = reg
            .validate_instance("IDTA 02006-2-0", &elements)
            .expect("found");
        // 1 matched out of 5 total
        assert!(result.conformance_score > 0.0);
        assert!(result.conformance_score < 1.0);
    }

    // ── TemplateVersion tests ─────────────────────────────────────────────

    #[test]
    fn test_version_display() {
        let v = TemplateVersion::new(1, 2, 3);
        assert_eq!(format!("{v}"), "1.2.3");
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = TemplateVersion::new(1, 0, 0);
        let v12 = TemplateVersion::new(1, 2, 0);
        let v2 = TemplateVersion::new(2, 0, 0);
        assert!(v1.is_compatible_with(&v12));
        assert!(!v1.is_compatible_with(&v2));
    }

    #[test]
    fn test_version_ordering() {
        let v1 = TemplateVersion::new(1, 0, 0);
        let v12 = TemplateVersion::new(1, 2, 0);
        let v2 = TemplateVersion::new(2, 0, 0);
        assert!(v1 < v12);
        assert!(v12 < v2);
    }

    // ── Display tests ─────────────────────────────────────────────────────

    #[test]
    fn test_element_type_display() {
        assert_eq!(format!("{}", ElementType::Property), "Property");
        assert_eq!(
            format!("{}", ElementType::Collection),
            "SubmodelElementCollection"
        );
        assert_eq!(format!("{}", ElementType::Entity), "Entity");
    }

    #[test]
    fn test_value_type_display() {
        assert_eq!(format!("{}", ValueType::String), "xs:string");
        assert_eq!(format!("{}", ValueType::Integer), "xs:integer");
        assert_eq!(
            format!("{}", ValueType::Custom("custom".to_string())),
            "custom"
        );
    }

    #[test]
    fn test_multiplicity_display() {
        assert_eq!(format!("{}", Multiplicity::One), "[1]");
        assert_eq!(format!("{}", Multiplicity::ZeroOrOne), "[0..1]");
        assert_eq!(format!("{}", Multiplicity::ZeroOrMore), "[0..*]");
        assert_eq!(format!("{}", Multiplicity::OneOrMore), "[1..*]");
    }

    #[test]
    fn test_multiplicity_is_required() {
        assert!(Multiplicity::One.is_required());
        assert!(Multiplicity::OneOrMore.is_required());
        assert!(!Multiplicity::ZeroOrOne.is_required());
        assert!(!Multiplicity::ZeroOrMore.is_required());
    }

    #[test]
    fn test_category_display() {
        assert_eq!(
            format!("{}", TemplateCategory::Identification),
            "Identification"
        );
        assert_eq!(format!("{}", TemplateCategory::TimeSeries), "Time Series");
    }

    // ── List all ──────────────────────────────────────────────────────────

    #[test]
    fn test_list_all() {
        let reg = std_registry();
        let all = reg.list_all();
        assert_eq!(all.len(), 8);
    }

    // ── Conditional required constraint ───────────────────────────────────

    #[test]
    fn test_conditional_required_constraint() {
        let mut reg = TemplateRegistry::new();
        reg.register(SubmodelTemplate {
            idta_id: "T-COND".to_string(),
            semantic_id: "https://ex.org/cond".to_string(),
            name: "Conditional Test".to_string(),
            version: TemplateVersion::new(1, 0, 0),
            description: "Test conditional constraints".to_string(),
            required_elements: vec![],
            optional_elements: vec![
                TemplateElement {
                    id_short: "A".to_string(),
                    semantic_id: None,
                    element_type: ElementType::Property,
                    value_type: Some(ValueType::String),
                    description: "Element A".to_string(),
                    multiplicity: Multiplicity::ZeroOrOne,
                    children: vec![],
                    example_value: None,
                },
                TemplateElement {
                    id_short: "B".to_string(),
                    semantic_id: None,
                    element_type: ElementType::Property,
                    value_type: Some(ValueType::String),
                    description: "Element B".to_string(),
                    multiplicity: Multiplicity::ZeroOrOne,
                    children: vec![],
                    example_value: None,
                },
            ],
            constraints: vec![TemplateConstraint::ConditionalRequired {
                condition_element: "A".to_string(),
                required_element: "B".to_string(),
            }],
            category: TemplateCategory::Other,
            tags: vec![],
        });

        // A present, B missing -> error
        let mut elements = HashMap::new();
        elements.insert("A".to_string(), "value".to_string());
        let result = reg.validate_instance("T-COND", &elements).expect("found");
        assert!(!result.is_valid);

        // A present, B present -> ok
        elements.insert("B".to_string(), "value".to_string());
        let result = reg.validate_instance("T-COND", &elements).expect("found");
        assert!(result.is_valid);
    }

    // ── Pattern constraint ────────────────────────────────────────────────

    #[test]
    fn test_pattern_constraint() {
        let mut reg = TemplateRegistry::new();
        reg.register(SubmodelTemplate {
            idta_id: "T-PAT".to_string(),
            semantic_id: "https://ex.org/pat".to_string(),
            name: "Pattern Test".to_string(),
            version: TemplateVersion::new(1, 0, 0),
            description: "Test pattern".to_string(),
            required_elements: vec![TemplateElement {
                id_short: "Email".to_string(),
                semantic_id: None,
                element_type: ElementType::Property,
                value_type: Some(ValueType::String),
                description: "Email".to_string(),
                multiplicity: Multiplicity::One,
                children: vec![],
                example_value: None,
            }],
            optional_elements: vec![],
            constraints: vec![TemplateConstraint::Pattern {
                element: "Email".to_string(),
                pattern: r"^[^@]+@[^@]+\.[^@]+$".to_string(),
            }],
            category: TemplateCategory::Other,
            tags: vec![],
        });

        let mut elements = HashMap::new();
        elements.insert("Email".to_string(), "user@example.com".to_string());
        let result = reg.validate_instance("T-PAT", &elements).expect("found");
        assert!(result.is_valid);

        elements.insert("Email".to_string(), "not-an-email".to_string());
        let result = reg.validate_instance("T-PAT", &elements).expect("found");
        assert!(!result.is_valid);
    }
}
