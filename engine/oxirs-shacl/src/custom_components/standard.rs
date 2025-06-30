//! Standard custom constraint component implementations
//!
//! This module provides a library of commonly useful constraint components
//! that extend SHACL with additional validation capabilities.

use crate::{ConstraintComponentId, Result, ShaclError};
use oxirs_core::model::Term;
use std::collections::HashMap;

use super::{
    constraints::CustomConstraint,
    metadata::{ComponentMetadata, ParameterConstraint, ParameterDefinition},
    CustomConstraintComponent,
};

/// Regular expression constraint component
#[derive(Debug)]
pub struct RegexConstraintComponent {
    component_id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl RegexConstraintComponent {
    pub fn new() -> Self {
        let component_id = ConstraintComponentId("ex:RegexConstraintComponent".to_string());
        let metadata = ComponentMetadata {
            name: "Regular Expression Constraint".to_string(),
            description: Some(
                "Validates text values against a regular expression pattern".to_string(),
            ),
            version: Some("1.0.0".to_string()),
            author: Some("OxiRS Team".to_string()),
            parameters: vec![
                ParameterDefinition {
                    name: "pattern".to_string(),
                    description: Some("Regular expression pattern to match".to_string()),
                    required: true,
                    datatype: Some("xsd:string".to_string()),
                    default_value: None,
                    validation_constraints: vec![
                        ParameterConstraint::MinLength(1),
                        ParameterConstraint::MaxLength(1000),
                    ],
                    cardinality: Some((1, Some(1))),
                    allowed_values: None,
                },
                ParameterDefinition {
                    name: "flags".to_string(),
                    description: Some("Regular expression flags (optional)".to_string()),
                    required: false,
                    datatype: Some("xsd:string".to_string()),
                    default_value: None,
                    validation_constraints: vec![ParameterConstraint::MaxLength(20)],
                    cardinality: Some((0, Some(1))),
                    allowed_values: Some(vec![
                        "i".to_string(),
                        "m".to_string(),
                        "s".to_string(),
                        "x".to_string(),
                        "im".to_string(),
                        "is".to_string(),
                        "ms".to_string(),
                    ]),
                },
            ],
            applicable_to_node_shapes: false,
            applicable_to_property_shapes: true,
            example: Some(
                r#"ex:EmailShape a sh:PropertyShape ;
    sh:path ex:email ;
    ex:pattern "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$" ."#
                    .to_string(),
            ),
        };

        Self {
            component_id,
            metadata,
        }
    }
}

impl CustomConstraintComponent for RegexConstraintComponent {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.component_id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, parameters: &HashMap<String, Term>) -> Result<()> {
        // Check required pattern parameter
        let pattern = parameters.get("pattern").ok_or_else(|| {
            ShaclError::Configuration("Pattern parameter is required".to_string())
        })?;

        // Validate that pattern is a literal
        if !matches!(pattern, Term::Literal(_)) {
            return Err(ShaclError::Configuration(
                "Pattern parameter must be a literal".to_string(),
            ));
        }

        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        self.validate_configuration(&parameters)?;

        Ok(CustomConstraint {
            component_id: self.component_id.clone(),
            parameters,
            sparql_query: None,
            validation_function: Some("regex".to_string()),
            message_template: Some("Value does not match required pattern".to_string()),
        })
    }
}

/// Range constraint component
#[derive(Debug)]
pub struct RangeConstraintComponent {
    component_id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl RangeConstraintComponent {
    pub fn new() -> Self {
        let component_id = ConstraintComponentId("ex:RangeConstraintComponent".to_string());
        let metadata = ComponentMetadata {
            name: "Numeric Range Constraint".to_string(),
            description: Some(
                "Validates that numeric values fall within a specified range".to_string(),
            ),
            version: Some("1.0.0".to_string()),
            author: Some("OxiRS Team".to_string()),
            parameters: vec![
                ParameterDefinition {
                    name: "minValue".to_string(),
                    description: Some("Minimum allowed value (inclusive)".to_string()),
                    required: false,
                    datatype: Some("xsd:decimal".to_string()),
                    default_value: None,
                    validation_constraints: vec![],
                    cardinality: Some((0, Some(1))),
                    allowed_values: None,
                },
                ParameterDefinition {
                    name: "maxValue".to_string(),
                    description: Some("Maximum allowed value (inclusive)".to_string()),
                    required: false,
                    datatype: Some("xsd:decimal".to_string()),
                    default_value: None,
                    validation_constraints: vec![],
                    cardinality: Some((0, Some(1))),
                    allowed_values: None,
                },
            ],
            applicable_to_node_shapes: false,
            applicable_to_property_shapes: true,
            example: Some(
                r#"ex:AgeShape a sh:PropertyShape ;
    sh:path ex:age ;
    ex:minValue 0 ;
    ex:maxValue 150 ."#
                    .to_string(),
            ),
        };

        Self {
            component_id,
            metadata,
        }
    }
}

impl CustomConstraintComponent for RangeConstraintComponent {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.component_id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, parameters: &HashMap<String, Term>) -> Result<()> {
        let has_min = parameters.contains_key("minValue");
        let has_max = parameters.contains_key("maxValue");

        if !has_min && !has_max {
            return Err(ShaclError::Configuration(
                "At least one of minValue or maxValue must be specified".to_string(),
            ));
        }

        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        self.validate_configuration(&parameters)?;

        Ok(CustomConstraint {
            component_id: self.component_id.clone(),
            parameters,
            sparql_query: None,
            validation_function: Some("range".to_string()),
            message_template: Some("Value is outside allowed range".to_string()),
        })
    }
}

/// URL validation constraint component
#[derive(Debug)]
pub struct UrlValidationComponent {
    component_id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl UrlValidationComponent {
    pub fn new() -> Self {
        let component_id = ConstraintComponentId("ex:UrlValidationComponent".to_string());
        let metadata = ComponentMetadata {
            name: "URL Validation Constraint".to_string(),
            description: Some("Validates that values are well-formed URLs".to_string()),
            version: Some("1.0.0".to_string()),
            author: Some("OxiRS Team".to_string()),
            parameters: vec![],
            applicable_to_node_shapes: false,
            applicable_to_property_shapes: true,
            example: Some(
                r#"ex:WebsiteShape a sh:PropertyShape ;
    sh:path ex:website ;
    ex:validUrl true ."#
                    .to_string(),
            ),
        };

        Self {
            component_id,
            metadata,
        }
    }
}

impl CustomConstraintComponent for UrlValidationComponent {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.component_id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, _parameters: &HashMap<String, Term>) -> Result<()> {
        // No required parameters
        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        Ok(CustomConstraint {
            component_id: self.component_id.clone(),
            parameters,
            sparql_query: None,
            validation_function: Some("url".to_string()),
            message_template: Some("Value is not a valid URL".to_string()),
        })
    }
}

/// Email validation constraint component
#[derive(Debug)]
pub struct EmailValidationComponent {
    component_id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl EmailValidationComponent {
    pub fn new() -> Self {
        let component_id = ConstraintComponentId("ex:EmailValidationComponent".to_string());
        let metadata = ComponentMetadata {
            name: "Email Validation Constraint".to_string(),
            description: Some(
                "Validates that literal values are well-formed email addresses".to_string(),
            ),
            version: Some("1.0.0".to_string()),
            author: Some("OxiRS Team".to_string()),
            parameters: vec![],
            applicable_to_node_shapes: false,
            applicable_to_property_shapes: true,
            example: Some(
                r#"ex:EmailShape a sh:PropertyShape ;
    sh:path ex:email ;
    ex:validEmail true ."#
                    .to_string(),
            ),
        };

        Self {
            component_id,
            metadata,
        }
    }
}

impl CustomConstraintComponent for EmailValidationComponent {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.component_id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, _parameters: &HashMap<String, Term>) -> Result<()> {
        // No required parameters
        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        Ok(CustomConstraint {
            component_id: self.component_id.clone(),
            parameters,
            sparql_query: None,
            validation_function: Some("email".to_string()),
            message_template: Some("Value is not a valid email address".to_string()),
        })
    }
}

/// SPARQL-based custom constraint component
#[derive(Debug)]
pub struct SparqlConstraintComponent {
    component_id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl SparqlConstraintComponent {
    pub fn new() -> Self {
        let component_id = ConstraintComponentId("ex:SparqlConstraintComponent".to_string());
        let metadata = ComponentMetadata {
            name: "SPARQL Custom Constraint".to_string(),
            description: Some(
                "Generic SPARQL-based constraint for custom validation logic".to_string(),
            ),
            version: Some("1.0.0".to_string()),
            author: Some("OxiRS Team".to_string()),
            parameters: vec![
                ParameterDefinition {
                    name: "query".to_string(),
                    description: Some("SPARQL ASK or SELECT query for validation".to_string()),
                    required: true,
                    datatype: Some("xsd:string".to_string()),
                    default_value: None,
                    validation_constraints: vec![
                        ParameterConstraint::MinLength(10),
                        ParameterConstraint::MaxLength(10000),
                    ],
                    cardinality: Some((1, Some(1))),
                    allowed_values: None,
                },
                ParameterDefinition {
                    name: "prefixes".to_string(),
                    description: Some("SPARQL prefixes to use with the query".to_string()),
                    required: false,
                    datatype: Some("xsd:string".to_string()),
                    default_value: None,
                    validation_constraints: vec![ParameterConstraint::MaxLength(5000)],
                    cardinality: Some((0, Some(1))),
                    allowed_values: None,
                },
            ],
            applicable_to_node_shapes: true,
            applicable_to_property_shapes: true,
            example: Some(
                r#"ex:CustomSparqlShape a sh:PropertyShape ;
    sh:path ex:customProperty ;
    ex:sparqlQuery "ASK { $this ex:hasRequiredRelation ?relation }" ."#
                    .to_string(),
            ),
        };

        Self {
            component_id,
            metadata,
        }
    }
}

impl CustomConstraintComponent for SparqlConstraintComponent {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.component_id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, parameters: &HashMap<String, Term>) -> Result<()> {
        // Check required query parameter
        let query = parameters
            .get("query")
            .ok_or_else(|| ShaclError::Configuration("Query parameter is required".to_string()))?;

        // Validate that query is a literal
        if !matches!(query, Term::Literal(_)) {
            return Err(ShaclError::Configuration(
                "Query parameter must be a literal".to_string(),
            ));
        }

        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        self.validate_configuration(&parameters)?;

        let query = parameters
            .get("query")
            .and_then(|t| match t {
                Term::Literal(lit) => Some(lit.value().to_string()),
                _ => None,
            })
            .unwrap();

        let prefixes = parameters.get("prefixes").and_then(|t| match t {
            Term::Literal(lit) => Some(lit.value().to_string()),
            _ => None,
        });

        let sparql_query = if let Some(prefixes) = prefixes {
            format!("{}\n{}", prefixes, query)
        } else {
            query
        };

        Ok(CustomConstraint {
            component_id: self.component_id.clone(),
            parameters,
            sparql_query: Some(sparql_query),
            validation_function: Some("sparql".to_string()),
            message_template: Some("Custom SPARQL constraint violated".to_string()),
        })
    }

    fn sparql_template(&self) -> Option<&str> {
        Some("ASK { $this ?customPredicate ?customValue }")
    }

    fn sparql_prefixes(&self) -> Option<&str> {
        Some("PREFIX ex: <http://example.org/>")
    }
}
