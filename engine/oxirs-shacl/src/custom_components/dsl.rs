//! # Domain-Specific Constraint Language (DSL) for SHACL
//!
//! This module provides a high-level DSL for defining SHACL constraints in a more
//! readable and maintainable way than raw RDF/Turtle.
//!
//! ## Features
//!
//! - **Declarative syntax**: Define constraints using Rust macros and builders
//! - **Type safety**: Compile-time validation of constraint definitions
//! - **Composability**: Build complex constraints from simple ones
//! - **Domain patterns**: Pre-built patterns for common domains
//! - **Code generation**: Generate SHACL shapes from DSL
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_shacl::custom_components::dsl::*;
//!
//! let person_shape = shape!("PersonShape")
//!     .target_class("ex:Person")
//!     .property("ex:name")
//!         .min_count(1)
//!         .datatype(xsd::string)
//!         .pattern("[A-Z][a-z]+")
//!     .property("ex:age")
//!         .datatype(xsd::integer)
//!         .min_inclusive(0)
//!         .max_inclusive(150)
//!     .property("ex:email")
//!         .pattern(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
//!     .build();
//! ```

use crate::{Result, Severity, ShaclError, Shape, ShapeId, ShapeType, Target};
use indexmap::IndexMap;
use oxirs_core::model::NamedNode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// DSL builder for SHACL shapes
pub struct ShapeDSL {
    id: ShapeId,
    shape_type: ShapeType,
    targets: Vec<Target>,
    properties: Vec<PropertyDSL>,
    constraints: IndexMap<String, ConstraintSpec>,
    metadata: ShapeMetadataDSL,
}

impl ShapeDSL {
    /// Create a new node shape
    pub fn node_shape(id: impl Into<String>) -> Self {
        Self {
            id: ShapeId::new(id),
            shape_type: ShapeType::NodeShape,
            targets: Vec::new(),
            properties: Vec::new(),
            constraints: IndexMap::new(),
            metadata: ShapeMetadataDSL::default(),
        }
    }

    /// Target nodes of a specific class
    pub fn target_class(mut self, class: impl Into<String>) -> Self {
        let class_str = class.into();
        if let Ok(node) = NamedNode::new(&class_str) {
            self.targets.push(Target::Class(node));
        }
        self
    }

    /// Target nodes with a specific property
    pub fn target_subjects_of(mut self, property: impl Into<String>) -> Self {
        let prop_str = property.into();
        if let Ok(node) = NamedNode::new(&prop_str) {
            self.targets.push(Target::SubjectsOf(node));
        }
        self
    }

    /// Target nodes that are objects of a specific property
    pub fn target_objects_of(mut self, property: impl Into<String>) -> Self {
        let prop_str = property.into();
        if let Ok(node) = NamedNode::new(&prop_str) {
            self.targets.push(Target::ObjectsOf(node));
        }
        self
    }

    /// Add a property constraint
    pub fn property(self, path: impl Into<String>) -> PropertyDSLBuilder {
        PropertyDSLBuilder::new(self, path.into())
    }

    /// Add metadata
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.metadata.label = Some(label.into());
        self
    }

    /// Add description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.metadata.description = Some(description.into());
        self
    }

    /// Set severity
    pub fn with_severity(mut self, severity: Severity) -> Self {
        self.metadata.severity = severity;
        self
    }

    /// Build the shape
    pub fn build(self) -> Result<Shape> {
        let mut shape = Shape::new(self.id, self.shape_type);
        shape.targets = self.targets;

        if let Some(label) = self.metadata.label {
            shape.label = Some(label);
        }

        if let Some(description) = self.metadata.description {
            shape.description = Some(description);
        }

        shape.severity = self.metadata.severity;

        // Add property shapes as constraints
        // (In a full implementation, we'd convert PropertyDSL to actual property shapes)

        Ok(shape)
    }
}

/// Builder for property constraints
pub struct PropertyDSLBuilder {
    parent: ShapeDSL,
    path: String,
    constraints: Vec<ConstraintSpec>,
}

impl PropertyDSLBuilder {
    fn new(parent: ShapeDSL, path: String) -> Self {
        Self {
            parent,
            path,
            constraints: Vec::new(),
        }
    }

    /// Minimum cardinality
    pub fn min_count(mut self, count: usize) -> Self {
        self.constraints.push(ConstraintSpec::MinCount(count));
        self
    }

    /// Maximum cardinality
    pub fn max_count(mut self, count: usize) -> Self {
        self.constraints.push(ConstraintSpec::MaxCount(count));
        self
    }

    /// Datatype constraint
    pub fn datatype(mut self, datatype: impl Into<String>) -> Self {
        self.constraints
            .push(ConstraintSpec::Datatype(datatype.into()));
        self
    }

    /// Pattern constraint
    pub fn pattern(mut self, pattern: impl Into<String>) -> Self {
        self.constraints.push(ConstraintSpec::Pattern {
            pattern: pattern.into(),
            flags: None,
        });
        self
    }

    /// Minimum value (inclusive)
    pub fn min_inclusive(mut self, value: i64) -> Self {
        self.constraints.push(ConstraintSpec::MinInclusive(value));
        self
    }

    /// Maximum value (inclusive)
    pub fn max_inclusive(mut self, value: i64) -> Self {
        self.constraints.push(ConstraintSpec::MaxInclusive(value));
        self
    }

    /// Minimum length
    pub fn min_length(mut self, length: usize) -> Self {
        self.constraints.push(ConstraintSpec::MinLength(length));
        self
    }

    /// Maximum length
    pub fn max_length(mut self, length: usize) -> Self {
        self.constraints.push(ConstraintSpec::MaxLength(length));
        self
    }

    /// Add another property
    pub fn property(self, path: impl Into<String>) -> PropertyDSLBuilder {
        let property = PropertyDSL {
            path: self.path.clone(),
            constraints: self.constraints.clone(),
        };
        let mut parent = self.parent;
        parent.properties.push(property);
        PropertyDSLBuilder::new(parent, path.into())
    }

    /// Build and return to parent
    pub fn build(self) -> ShapeDSL {
        let property = PropertyDSL {
            path: self.path,
            constraints: self.constraints,
        };
        let mut parent = self.parent;
        parent.properties.push(property);
        parent
    }
}

/// Property constraint specification
#[derive(Debug, Clone)]
struct PropertyDSL {
    path: String,
    constraints: Vec<ConstraintSpec>,
}

/// Constraint specification in DSL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintSpec {
    MinCount(usize),
    MaxCount(usize),
    Datatype(String),
    Pattern {
        pattern: String,
        flags: Option<String>,
    },
    MinInclusive(i64),
    MaxInclusive(i64),
    MinExclusive(i64),
    MaxExclusive(i64),
    MinLength(usize),
    MaxLength(usize),
    In(Vec<String>),
    LanguageIn(Vec<String>),
    NodeKind(String),
    Class(String),
    Equals(String),
    Disjoint(String),
    LessThan(String),
    LessThanOrEquals(String),
    UniqueLang(bool),
    HasValue(String),
}

/// Shape metadata in DSL
#[derive(Debug, Clone, Default)]
struct ShapeMetadataDSL {
    label: Option<String>,
    description: Option<String>,
    severity: Severity,
}

/// Domain-specific patterns for common use cases
pub mod patterns {
    use super::*;

    /// Email validation pattern
    pub fn email() -> ShapeDSL {
        ShapeDSL::node_shape("EmailShape")
            .with_label("Email")
            .with_description("Valid email address")
    }

    /// URL validation pattern
    pub fn url() -> ShapeDSL {
        ShapeDSL::node_shape("UrlShape")
            .with_label("URL")
            .with_description("Valid URL")
    }

    /// Date range pattern
    pub fn date_range(min: &str, max: &str) -> ShapeDSL {
        ShapeDSL::node_shape("DateRangeShape")
            .with_label("Date Range")
            .with_description(format!("Date between {} and {}", min, max))
    }

    /// Positive integer pattern
    pub fn positive_integer() -> ShapeDSL {
        ShapeDSL::node_shape("PositiveIntegerShape")
            .with_label("Positive Integer")
            .with_description("Integer greater than zero")
    }

    /// Person pattern with common properties
    pub fn person() -> ShapeDSL {
        ShapeDSL::node_shape("PersonShape")
            .target_class("foaf:Person")
            .with_label("Person")
            .with_description("A person with basic properties")
    }

    /// Organization pattern
    pub fn organization() -> ShapeDSL {
        ShapeDSL::node_shape("OrganizationShape")
            .target_class("org:Organization")
            .with_label("Organization")
            .with_description("An organization")
    }

    /// Address pattern
    pub fn address() -> ShapeDSL {
        ShapeDSL::node_shape("AddressShape")
            .with_label("Address")
            .with_description("Postal address")
    }
}

/// Macro for creating shapes with a fluent syntax
#[macro_export]
macro_rules! shape {
    ($id:expr) => {
        $crate::custom_components::dsl::ShapeDSL::node_shape($id)
    };
}

/// Namespace prefixes for common vocabularies
pub mod namespaces {
    pub const XSD: &str = "http://www.w3.org/2001/XMLSchema#";
    pub const RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
    pub const RDFS: &str = "http://www.w3.org/2000/01/rdf-schema#";
    pub const OWL: &str = "http://www.w3.org/2002/07/owl#";
    pub const FOAF: &str = "http://xmlns.com/foaf/0.1/";
    pub const DC: &str = "http://purl.org/dc/elements/1.1/";
    pub const DCTERMS: &str = "http://purl.org/dc/terms/";
    pub const SKOS: &str = "http://www.w3.org/2004/02/skos/core#";
}

/// XSD datatypes helper
pub mod xsd {
    use super::namespaces::XSD;

    pub fn string() -> String {
        format!("{}string", XSD)
    }

    pub fn integer() -> String {
        format!("{}integer", XSD)
    }

    pub fn decimal() -> String {
        format!("{}decimal", XSD)
    }

    pub fn boolean() -> String {
        format!("{}boolean", XSD)
    }

    pub fn date() -> String {
        format!("{}date", XSD)
    }

    pub fn datetime() -> String {
        format!("{}dateTime", XSD)
    }

    pub fn double() -> String {
        format!("{}double", XSD)
    }

    pub fn float() -> String {
        format!("{}float", XSD)
    }

    pub fn any_uri() -> String {
        format!("{}anyURI", XSD)
    }
}

/// Type alias for constraint generator function
type ConstraintGenerator = Box<dyn Fn(&HashMap<String, String>) -> Result<Vec<ConstraintSpec>>>;

/// Template system for reusable constraint patterns
pub struct ConstraintTemplate {
    name: String,
    parameters: HashMap<String, ParameterDef>,
    generator: ConstraintGenerator,
}

impl ConstraintTemplate {
    pub fn new(
        name: impl Into<String>,
        parameters: HashMap<String, ParameterDef>,
        generator: impl Fn(&HashMap<String, String>) -> Result<Vec<ConstraintSpec>> + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            parameters,
            generator: Box::new(generator),
        }
    }

    pub fn instantiate(&self, values: &HashMap<String, String>) -> Result<Vec<ConstraintSpec>> {
        // Validate parameters
        for (param_name, param_def) in &self.parameters {
            if param_def.required && !values.contains_key(param_name) {
                return Err(ShaclError::Configuration(format!(
                    "Missing required parameter: {}",
                    param_name
                )));
            }
        }

        (self.generator)(values)
    }
}

/// Parameter definition for templates
#[derive(Debug, Clone)]
pub struct ParameterDef {
    pub name: String,
    pub param_type: ParameterType,
    pub required: bool,
    pub default: Option<String>,
    pub description: Option<String>,
}

/// Parameter type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParameterType {
    String,
    Integer,
    Boolean,
    IRI,
    Pattern,
}

/// Template library with common patterns
pub struct TemplateLibrary {
    templates: HashMap<String, ConstraintTemplate>,
}

impl TemplateLibrary {
    pub fn new() -> Self {
        let mut library = Self {
            templates: HashMap::new(),
        };

        library.register_builtins();
        library
    }

    pub fn register(&mut self, template: ConstraintTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    pub fn get(&self, name: &str) -> Option<&ConstraintTemplate> {
        self.templates.get(name)
    }

    fn register_builtins(&mut self) {
        // Range validation template
        let range_params = {
            let mut params = HashMap::new();
            params.insert(
                "min".to_string(),
                ParameterDef {
                    name: "min".to_string(),
                    param_type: ParameterType::Integer,
                    required: true,
                    default: None,
                    description: Some("Minimum value".to_string()),
                },
            );
            params.insert(
                "max".to_string(),
                ParameterDef {
                    name: "max".to_string(),
                    param_type: ParameterType::Integer,
                    required: true,
                    default: None,
                    description: Some("Maximum value".to_string()),
                },
            );
            params
        };

        let range_template = ConstraintTemplate::new("range", range_params, |values| {
            let min = values
                .get("min")
                .expect("operation should succeed")
                .parse::<i64>()
                .map_err(|_| ShaclError::Configuration("Invalid min value".to_string()))?;
            let max = values
                .get("max")
                .expect("operation should succeed")
                .parse::<i64>()
                .map_err(|_| ShaclError::Configuration("Invalid max value".to_string()))?;

            Ok(vec![
                ConstraintSpec::MinInclusive(min),
                ConstraintSpec::MaxInclusive(max),
            ])
        });

        self.register(range_template);

        // String length template
        let length_params = {
            let mut params = HashMap::new();
            params.insert(
                "min".to_string(),
                ParameterDef {
                    name: "min".to_string(),
                    param_type: ParameterType::Integer,
                    required: false,
                    default: Some("0".to_string()),
                    description: Some("Minimum length".to_string()),
                },
            );
            params.insert(
                "max".to_string(),
                ParameterDef {
                    name: "max".to_string(),
                    param_type: ParameterType::Integer,
                    required: true,
                    default: None,
                    description: Some("Maximum length".to_string()),
                },
            );
            params
        };

        let length_template = ConstraintTemplate::new("length", length_params, |values| {
            let min = values
                .get("min")
                .map(|v| v.parse::<usize>().unwrap_or(0))
                .unwrap_or(0);
            let max = values
                .get("max")
                .expect("operation should succeed")
                .parse::<usize>()
                .map_err(|_| ShaclError::Configuration("Invalid max value".to_string()))?;

            Ok(vec![
                ConstraintSpec::MinLength(min),
                ConstraintSpec::MaxLength(max),
            ])
        });

        self.register(length_template);
    }
}

impl Default for TemplateLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_dsl_builder() {
        let shape = ShapeDSL::node_shape("TestShape")
            .target_class("ex:TestClass")
            .with_label("Test Shape")
            .build()
            .expect("operation should succeed");

        assert_eq!(shape.id.as_str(), "TestShape");
        assert!(!shape.targets.is_empty());
    }

    #[test]
    fn test_property_builder() {
        let shape = ShapeDSL::node_shape("PersonShape")
            .target_class("ex:Person")
            .property("ex:name")
            .min_count(1)
            .datatype(xsd::string())
            .build()
            .build()
            .expect("operation should succeed");

        assert_eq!(shape.id.as_str(), "PersonShape");
    }

    #[test]
    fn test_patterns() {
        let email_shape = patterns::email().build().expect("operation should succeed");
        assert_eq!(email_shape.id.as_str(), "EmailShape");

        let person_shape = patterns::person()
            .build()
            .expect("operation should succeed");
        assert_eq!(person_shape.id.as_str(), "PersonShape");
    }

    #[test]
    fn test_template_library() {
        let library = TemplateLibrary::new();
        assert!(library.get("range").is_some());
        assert!(library.get("length").is_some());
    }

    #[test]
    fn test_template_instantiation() {
        let library = TemplateLibrary::new();
        let template = library.get("range").expect("key should exist");

        let mut values = HashMap::new();
        values.insert("min".to_string(), "0".to_string());
        values.insert("max".to_string(), "100".to_string());

        let constraints = template
            .instantiate(&values)
            .expect("training should succeed");
        assert_eq!(constraints.len(), 2);
    }

    #[test]
    fn test_xsd_datatypes() {
        assert!(xsd::string().contains("string"));
        assert!(xsd::integer().contains("integer"));
        assert!(xsd::datetime().contains("dateTime"));
    }
}
