//! Component metadata and parameter definitions
//!
//! This module defines the metadata structures used to describe custom constraint components,
//! including parameter definitions, validation rules, and component information.

use crate::{ConstraintComponentId, Severity};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metadata for a constraint component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetadata {
    /// Component name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Version
    pub version: Option<String>,
    /// Author
    pub author: Option<String>,
    /// Parameters this component accepts
    pub parameters: Vec<ParameterDefinition>,
    /// Whether this component can be used on node shapes
    pub applicable_to_node_shapes: bool,
    /// Whether this component can be used on property shapes
    pub applicable_to_property_shapes: bool,
    /// Example usage
    pub example: Option<String>,
}

/// Parameter definition for a custom constraint component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDefinition {
    /// Parameter name
    pub name: String,
    /// Parameter description
    pub description: Option<String>,
    /// Whether this parameter is required
    pub required: bool,
    /// Expected data type
    pub datatype: Option<String>,
    /// Default value
    pub default_value: Option<String>,
    /// Parameter validation constraints
    pub validation_constraints: Vec<ParameterConstraint>,
    /// Parameter cardinality (min, max)
    pub cardinality: Option<(u32, Option<u32>)>,
    /// Allowed values enumeration
    pub allowed_values: Option<Vec<String>>,
}

/// Component library for organizing related components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentLibrary {
    /// Library identifier
    pub id: String,
    /// Library name
    pub name: String,
    /// Library description
    pub description: Option<String>,
    /// Library version
    pub version: String,
    /// Library author
    pub author: Option<String>,
    /// Components in this library
    pub components: Vec<ConstraintComponentId>,
    /// Library dependencies
    pub dependencies: Vec<String>,
    /// Library metadata
    pub metadata: HashMap<String, String>,
}

/// Validation rule for parameter validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: Option<String>,
    /// Rule type
    pub rule_type: ValidationRuleType,
    /// Rule severity
    pub severity: Severity,
    /// Rule condition
    pub condition: ValidationCondition,
}

/// Validation rule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    Required,
    DataType,
    Range,
    Pattern,
    Enum,
    Custom,
}

/// Validation condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationCondition {
    Required,
    DataTypeMatch(String),
    RangeCheck { min: Option<f64>, max: Option<f64> },
    PatternMatch(String),
    EnumValues(Vec<String>),
    CustomFunction(String),
}

/// Parameter constraint for enhanced validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterConstraint {
    /// Minimum length for string parameters
    MinLength(u32),
    /// Maximum length for string parameters
    MaxLength(u32),
    /// Regular expression pattern
    Pattern(String),
    /// Numeric range constraint
    Range { min: Option<f64>, max: Option<f64> },
    /// Custom validation function
    CustomValidator(String),
}
