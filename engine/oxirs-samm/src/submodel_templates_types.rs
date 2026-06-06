//! Submodel Templates — Types
//!
//! Template structs, variable bindings, constraint types, and instantiation
//! configuration for the AAS submodel template library.

use serde::{Deserialize, Serialize};
use std::fmt;

// ── Template types ────────────────────────────────────────────────────────────

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
    Custom(std::string::String),
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

// ── Validation result types ───────────────────────────────────────────────────

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
