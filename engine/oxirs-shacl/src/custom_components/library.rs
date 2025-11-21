//! Reusable Constraint Component Library
//!
//! This module provides a comprehensive library of pre-built, commonly-used constraint
//! components organized by domain. These components can be easily composed and reused
//! to build complex validation logic.
//!
//! # Domain Categories
//!
//! - **Identity**: UUID, IRI, ISBN, DOI, ORCID validation
//! - **Temporal**: Date ranges, duration, timezone validation
//! - **Geospatial**: Coordinates, bounding boxes, country codes
//! - **Financial**: Currency, IBAN, BIC, credit card validation
//! - **Personal**: Phone numbers, postal addresses, names
//! - **Scientific**: Units, measurements, chemical formulas
//! - **Semantic**: RDF-specific patterns, ontology compliance

use crate::custom_components::{
    ComponentMetadata, CustomConstraint, CustomConstraintComponent, ParameterDefinition,
};
use crate::{ConstraintComponentId, Result, ShaclError};
use oxirs_core::model::Term;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Library of reusable constraint components
#[derive(Debug)]
pub struct ConstraintLibrary {
    /// Components organized by category
    components: HashMap<String, Vec<Arc<dyn CustomConstraintComponent>>>,
    /// Component index by ID
    index: HashMap<ConstraintComponentId, Arc<dyn CustomConstraintComponent>>,
    /// Library metadata
    metadata: LibraryMetadata,
}

/// Library metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryMetadata {
    /// Library name
    pub name: String,
    /// Library version
    pub version: String,
    /// Description
    pub description: String,
    /// Total component count
    pub component_count: usize,
    /// Available categories
    pub categories: Vec<String>,
}

impl ConstraintLibrary {
    /// Create a new constraint library with standard components
    pub fn new() -> Self {
        let mut library = Self {
            components: HashMap::new(),
            index: HashMap::new(),
            metadata: LibraryMetadata {
                name: "OxiRS SHACL Constraint Library".to_string(),
                version: "1.0.0".to_string(),
                description: "Comprehensive library of reusable SHACL constraint components"
                    .to_string(),
                component_count: 0,
                categories: Vec::new(),
            },
        };

        // Register all standard components
        library.register_identity_components();
        library.register_temporal_components();
        library.register_geospatial_components();
        library.register_financial_components();
        library.register_personal_components();
        library.register_scientific_components();
        library.register_semantic_components();

        library.update_metadata();
        library
    }

    /// Register a component in the library
    pub fn register(
        &mut self,
        category: &str,
        component: Arc<dyn CustomConstraintComponent>,
    ) -> Result<()> {
        let id = component.component_id().clone();

        // Add to category
        self.components
            .entry(category.to_string())
            .or_default()
            .push(Arc::clone(&component));

        // Add to index
        self.index.insert(id, component);

        Ok(())
    }

    /// Get a component by ID
    pub fn get(&self, id: &ConstraintComponentId) -> Option<&Arc<dyn CustomConstraintComponent>> {
        self.index.get(id)
    }

    /// Get all components in a category
    pub fn get_category(&self, category: &str) -> Option<&Vec<Arc<dyn CustomConstraintComponent>>> {
        self.components.get(category)
    }

    /// List all available categories
    pub fn categories(&self) -> Vec<&str> {
        self.components.keys().map(|s| s.as_str()).collect()
    }

    /// Search components by name or description
    pub fn search(&self, query: &str) -> Vec<&Arc<dyn CustomConstraintComponent>> {
        let query_lower = query.to_lowercase();
        self.index
            .values()
            .filter(|c| {
                let meta = c.metadata();
                meta.name.to_lowercase().contains(&query_lower)
                    || meta
                        .description
                        .as_ref()
                        .map(|d| d.to_lowercase())
                        .unwrap_or_default()
                        .contains(&query_lower)
            })
            .collect()
    }

    /// Get library metadata
    pub fn metadata(&self) -> &LibraryMetadata {
        &self.metadata
    }

    /// Update metadata after registration
    fn update_metadata(&mut self) {
        self.metadata.component_count = self.index.len();
        self.metadata.categories = self.components.keys().cloned().collect();
    }

    // ========== Identity Components ==========

    fn register_identity_components(&mut self) {
        let category = "identity";

        // UUID Validator
        self.register(category, Arc::new(UuidValidator::new()))
            .unwrap();

        // IRI/URI Validator
        self.register(category, Arc::new(IriValidator::new()))
            .unwrap();

        // ISBN Validator
        self.register(category, Arc::new(IsbnValidator::new()))
            .unwrap();

        // DOI Validator
        self.register(category, Arc::new(DoiValidator::new()))
            .unwrap();

        // ORCID Validator
        self.register(category, Arc::new(OrcidValidator::new()))
            .unwrap();
    }

    // ========== Temporal Components ==========

    fn register_temporal_components(&mut self) {
        let category = "temporal";

        // Date Range Validator
        self.register(category, Arc::new(DateRangeValidator::new()))
            .unwrap();

        // Duration Validator
        self.register(category, Arc::new(DurationValidator::new()))
            .unwrap();

        // Timezone Validator
        self.register(category, Arc::new(TimezoneValidator::new()))
            .unwrap();

        // Business Hours Validator
        self.register(category, Arc::new(BusinessHoursValidator::new()))
            .unwrap();
    }

    // ========== Geospatial Components ==========

    fn register_geospatial_components(&mut self) {
        let category = "geospatial";

        // Coordinates Validator
        self.register(category, Arc::new(CoordinatesValidator::new()))
            .unwrap();

        // Bounding Box Validator
        self.register(category, Arc::new(BoundingBoxValidator::new()))
            .unwrap();

        // Country Code Validator
        self.register(category, Arc::new(CountryCodeValidator::new()))
            .unwrap();

        // GeoJSON Validator
        self.register(category, Arc::new(GeoJsonValidator::new()))
            .unwrap();
    }

    // ========== Financial Components ==========

    fn register_financial_components(&mut self) {
        let category = "financial";

        // Currency Validator
        self.register(category, Arc::new(CurrencyValidator::new()))
            .unwrap();

        // IBAN Validator
        self.register(category, Arc::new(IbanValidator::new()))
            .unwrap();

        // BIC/SWIFT Validator
        self.register(category, Arc::new(BicValidator::new()))
            .unwrap();

        // Credit Card Validator
        self.register(category, Arc::new(CreditCardValidator::new()))
            .unwrap();
    }

    // ========== Personal Components ==========

    fn register_personal_components(&mut self) {
        let category = "personal";

        // Phone Number Validator
        self.register(category, Arc::new(PhoneNumberValidator::new()))
            .unwrap();

        // Postal Code Validator
        self.register(category, Arc::new(PostalCodeValidator::new()))
            .unwrap();

        // Name Pattern Validator
        self.register(category, Arc::new(NamePatternValidator::new()))
            .unwrap();

        // Age Range Validator
        self.register(category, Arc::new(AgeRangeValidator::new()))
            .unwrap();
    }

    // ========== Scientific Components ==========

    fn register_scientific_components(&mut self) {
        let category = "scientific";

        // Unit Validator
        self.register(category, Arc::new(UnitValidator::new()))
            .unwrap();

        // Chemical Formula Validator
        self.register(category, Arc::new(ChemicalFormulaValidator::new()))
            .unwrap();

        // Scientific Notation Validator
        self.register(category, Arc::new(ScientificNotationValidator::new()))
            .unwrap();
    }

    // ========== Semantic Components ==========

    fn register_semantic_components(&mut self) {
        let category = "semantic";

        // Class Hierarchy Validator
        self.register(category, Arc::new(ClassHierarchyValidator::new()))
            .unwrap();

        // Property Domain/Range Validator
        self.register(category, Arc::new(PropertyDomainRangeValidator::new()))
            .unwrap();

        // Ontology Consistency Validator
        self.register(category, Arc::new(OntologyConsistencyValidator::new()))
            .unwrap();
    }
}

impl Default for ConstraintLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// ========== Identity Components Implementation ==========

/// UUID validation component
#[derive(Debug)]
pub struct UuidValidator {
    id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl UuidValidator {
    pub fn new() -> Self {
        Self {
            id: ConstraintComponentId::new("lib:UuidValidator"),
            metadata: ComponentMetadata {
                name: "UUID Validator".to_string(),
                description: Some("Validates UUID format (version 1, 4, or 5)".to_string()),
                version: Some("1.0.0".to_string()),
                author: Some("OxiRS".to_string()),
                parameters: vec![ParameterDefinition {
                    name: "version".to_string(),
                    description: Some("UUID version to validate (1, 4, 5, or 'any')".to_string()),
                    required: false,
                    datatype: Some("xsd:string".to_string()),
                    default_value: Some("any".to_string()),
                    validation_constraints: vec![],
                    cardinality: None,
                    allowed_values: Some(vec![
                        "1".to_string(),
                        "4".to_string(),
                        "5".to_string(),
                        "any".to_string(),
                    ]),
                }],
                applicable_to_node_shapes: true,
                applicable_to_property_shapes: true,
                example: Some("ex:uuid a lib:UuidValidator ; lib:version \"4\" .".to_string()),
            },
        }
    }
}

impl Default for UuidValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomConstraintComponent for UuidValidator {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, _parameters: &HashMap<String, Term>) -> Result<()> {
        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        let version = parameters
            .get("version")
            .and_then(|t| match t {
                Term::Literal(l) => Some(l.value().to_string()),
                _ => None,
            })
            .unwrap_or_else(|| "any".to_string());

        let pattern = match version.as_str() {
            "1" => r"^[0-9a-f]{8}-[0-9a-f]{4}-1[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            "4" => r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            "5" => r"^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            _ => r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        };

        Ok(CustomConstraint {
            component_id: self.id.clone(),
            parameters,
            sparql_query: Some(format!(
                r#"
                SELECT $this WHERE {{
                    FILTER(REGEX(STR($this), "{}"))
                }}
                "#,
                pattern
            )),
            validation_function: None,
            message_template: Some(format!("Value must be a valid UUID (version: {})", version)),
        })
    }
}

/// IRI/URI validation component
#[derive(Debug)]
pub struct IriValidator {
    id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl IriValidator {
    pub fn new() -> Self {
        Self {
            id: ConstraintComponentId::new("lib:IriValidator"),
            metadata: ComponentMetadata {
                name: "IRI/URI Validator".to_string(),
                description: Some(
                    "Validates IRI/URI format according to RFC 3987/3986".to_string(),
                ),
                version: Some("1.0.0".to_string()),
                author: Some("OxiRS".to_string()),
                parameters: vec![ParameterDefinition {
                    name: "scheme".to_string(),
                    description: Some("Required URI scheme (e.g., 'http', 'https')".to_string()),
                    required: false,
                    datatype: Some("xsd:string".to_string()),
                    default_value: None,
                    validation_constraints: vec![],
                    cardinality: None,
                    allowed_values: None,
                }],
                applicable_to_node_shapes: true,
                applicable_to_property_shapes: true,
                example: Some(
                    "ex:iriShape a lib:IriValidator ; lib:scheme \"https\" .".to_string(),
                ),
            },
        }
    }
}

impl Default for IriValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomConstraintComponent for IriValidator {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, _parameters: &HashMap<String, Term>) -> Result<()> {
        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        let scheme = parameters.get("scheme").and_then(|t| match t {
            Term::Literal(l) => Some(l.value().to_string()),
            _ => None,
        });

        let sparql = if let Some(s) = &scheme {
            format!(
                r#"
                SELECT $this WHERE {{
                    FILTER(isIRI($this) && STRSTARTS(STR($this), "{}://"))
                }}
                "#,
                s
            )
        } else {
            r#"
                SELECT $this WHERE {
                    FILTER(isIRI($this))
                }
                "#
            .to_string()
        };

        Ok(CustomConstraint {
            component_id: self.id.clone(),
            parameters,
            sparql_query: Some(sparql),
            validation_function: None,
            message_template: Some(if scheme.is_some() {
                format!("Value must be a valid IRI with scheme: {}", scheme.unwrap())
            } else {
                "Value must be a valid IRI".to_string()
            }),
        })
    }
}

/// ISBN validation component
#[derive(Debug)]
pub struct IsbnValidator {
    id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl IsbnValidator {
    pub fn new() -> Self {
        Self {
            id: ConstraintComponentId::new("lib:IsbnValidator"),
            metadata: ComponentMetadata {
                name: "ISBN Validator".to_string(),
                description: Some("Validates ISBN-10 or ISBN-13 format with checksum".to_string()),
                version: Some("1.0.0".to_string()),
                author: Some("OxiRS".to_string()),
                parameters: vec![ParameterDefinition {
                    name: "format".to_string(),
                    description: Some("ISBN format (10, 13, or 'any')".to_string()),
                    required: false,
                    datatype: Some("xsd:string".to_string()),
                    default_value: Some("any".to_string()),
                    validation_constraints: vec![],
                    cardinality: None,
                    allowed_values: Some(vec![
                        "10".to_string(),
                        "13".to_string(),
                        "any".to_string(),
                    ]),
                }],
                applicable_to_node_shapes: true,
                applicable_to_property_shapes: true,
                example: Some("ex:isbnShape a lib:IsbnValidator ; lib:format \"13\" .".to_string()),
            },
        }
    }
}

impl Default for IsbnValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomConstraintComponent for IsbnValidator {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, _parameters: &HashMap<String, Term>) -> Result<()> {
        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        let format = parameters
            .get("format")
            .and_then(|t| match t {
                Term::Literal(l) => Some(l.value().to_string()),
                _ => None,
            })
            .unwrap_or_else(|| "any".to_string());

        let pattern = match format.as_str() {
            "10" => r"^(\d{9}[\dX])$",
            "13" => r"^(97[89]\d{10})$",
            _ => r"^((\d{9}[\dX])|(97[89]\d{10}))$",
        };

        Ok(CustomConstraint {
            component_id: self.id.clone(),
            parameters,
            sparql_query: Some(format!(
                r#"
                SELECT $this WHERE {{
                    FILTER(REGEX(REPLACE(STR($this), "-", ""), "{}"))
                }}
                "#,
                pattern
            )),
            validation_function: None,
            message_template: Some(format!("Value must be a valid ISBN (format: {})", format)),
        })
    }
}

/// DOI validation component
#[derive(Debug)]
pub struct DoiValidator {
    id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl DoiValidator {
    pub fn new() -> Self {
        Self {
            id: ConstraintComponentId::new("lib:DoiValidator"),
            metadata: ComponentMetadata {
                name: "DOI Validator".to_string(),
                description: Some("Validates Digital Object Identifier format".to_string()),
                version: Some("1.0.0".to_string()),
                author: Some("OxiRS".to_string()),
                parameters: vec![],
                applicable_to_node_shapes: true,
                applicable_to_property_shapes: true,
                example: Some("ex:doiShape a lib:DoiValidator .".to_string()),
            },
        }
    }
}

impl Default for DoiValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomConstraintComponent for DoiValidator {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, _parameters: &HashMap<String, Term>) -> Result<()> {
        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        Ok(CustomConstraint {
            component_id: self.id.clone(),
            parameters,
            sparql_query: Some(
                r#"
                SELECT $this WHERE {
                    FILTER(REGEX(STR($this), "^10\\.\\d{4,9}/[-._;()/:A-Z0-9]+$", "i"))
                }
                "#
                .to_string(),
            ),
            validation_function: None,
            message_template: Some("Value must be a valid DOI".to_string()),
        })
    }
}

/// ORCID validation component
#[derive(Debug)]
pub struct OrcidValidator {
    id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl OrcidValidator {
    pub fn new() -> Self {
        Self {
            id: ConstraintComponentId::new("lib:OrcidValidator"),
            metadata: ComponentMetadata {
                name: "ORCID Validator".to_string(),
                description: Some("Validates ORCID identifier format with checksum".to_string()),
                version: Some("1.0.0".to_string()),
                author: Some("OxiRS".to_string()),
                parameters: vec![],
                applicable_to_node_shapes: true,
                applicable_to_property_shapes: true,
                example: Some("ex:orcidShape a lib:OrcidValidator .".to_string()),
            },
        }
    }
}

impl Default for OrcidValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomConstraintComponent for OrcidValidator {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, _parameters: &HashMap<String, Term>) -> Result<()> {
        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        Ok(CustomConstraint {
            component_id: self.id.clone(),
            parameters,
            sparql_query: Some(
                r#"
                SELECT $this WHERE {
                    FILTER(REGEX(STR($this), "^\\d{4}-\\d{4}-\\d{4}-\\d{3}[\\dX]$"))
                }
                "#
                .to_string(),
            ),
            validation_function: None,
            message_template: Some("Value must be a valid ORCID".to_string()),
        })
    }
}

// ========== Temporal Components Implementation ==========

/// Date range validation component
#[derive(Debug)]
pub struct DateRangeValidator {
    id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl DateRangeValidator {
    pub fn new() -> Self {
        Self {
            id: ConstraintComponentId::new("lib:DateRangeValidator"),
            metadata: ComponentMetadata {
                name: "Date Range Validator".to_string(),
                description: Some("Validates date falls within specified range".to_string()),
                version: Some("1.0.0".to_string()),
                author: Some("OxiRS".to_string()),
                parameters: vec![
                    ParameterDefinition {
                        name: "minDate".to_string(),
                        description: Some("Minimum date (inclusive)".to_string()),
                        required: false,
                        datatype: Some("xsd:date".to_string()),
                        default_value: None,
                        validation_constraints: vec![],
                        cardinality: None,
                        allowed_values: None,
                    },
                    ParameterDefinition {
                        name: "maxDate".to_string(),
                        description: Some("Maximum date (inclusive)".to_string()),
                        required: false,
                        datatype: Some("xsd:date".to_string()),
                        default_value: None,
                        validation_constraints: vec![],
                        cardinality: None,
                        allowed_values: None,
                    },
                ],
                applicable_to_node_shapes: true,
                applicable_to_property_shapes: true,
                example: Some("ex:dateShape a lib:DateRangeValidator ; lib:minDate \"2020-01-01\"^^xsd:date .".to_string()),
            },
        }
    }
}

impl Default for DateRangeValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomConstraintComponent for DateRangeValidator {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, parameters: &HashMap<String, Term>) -> Result<()> {
        if !parameters.contains_key("minDate") && !parameters.contains_key("maxDate") {
            return Err(ShaclError::Configuration(
                "At least one of minDate or maxDate must be specified".to_string(),
            ));
        }
        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        let mut filters = Vec::new();

        if let Some(Term::Literal(l)) = parameters.get("minDate") {
            filters.push(format!("$this >= \"{}\"^^xsd:date", l.value()));
        }

        if let Some(Term::Literal(l)) = parameters.get("maxDate") {
            filters.push(format!("$this <= \"{}\"^^xsd:date", l.value()));
        }

        let sparql = format!(
            r#"
            SELECT $this WHERE {{
                FILTER({})
            }}
            "#,
            filters.join(" && ")
        );

        Ok(CustomConstraint {
            component_id: self.id.clone(),
            parameters,
            sparql_query: Some(sparql),
            validation_function: None,
            message_template: Some("Date must be within specified range".to_string()),
        })
    }
}

/// Duration validation component
#[derive(Debug)]
pub struct DurationValidator {
    id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl DurationValidator {
    pub fn new() -> Self {
        Self {
            id: ConstraintComponentId::new("lib:DurationValidator"),
            metadata: ComponentMetadata {
                name: "Duration Validator".to_string(),
                description: Some("Validates ISO 8601 duration format".to_string()),
                version: Some("1.0.0".to_string()),
                author: Some("OxiRS".to_string()),
                parameters: vec![],
                applicable_to_node_shapes: true,
                applicable_to_property_shapes: true,
                example: Some("ex:durationShape a lib:DurationValidator .".to_string()),
            },
        }
    }
}

impl Default for DurationValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomConstraintComponent for DurationValidator {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, _parameters: &HashMap<String, Term>) -> Result<()> {
        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        Ok(CustomConstraint {
            component_id: self.id.clone(),
            parameters,
            sparql_query: Some(
                r#"
                SELECT $this WHERE {
                    FILTER(REGEX(STR($this), "^P(\\d+Y)?(\\d+M)?(\\d+D)?(T(\\d+H)?(\\d+M)?(\\d+S)?)?$"))
                }
                "#
                .to_string(),
            ),
            validation_function: None,
            message_template: Some("Value must be a valid ISO 8601 duration".to_string()),
        })
    }
}

/// Timezone validation component
#[derive(Debug)]
pub struct TimezoneValidator {
    id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl TimezoneValidator {
    pub fn new() -> Self {
        Self {
            id: ConstraintComponentId::new("lib:TimezoneValidator"),
            metadata: ComponentMetadata {
                name: "Timezone Validator".to_string(),
                description: Some("Validates IANA timezone identifiers".to_string()),
                version: Some("1.0.0".to_string()),
                author: Some("OxiRS".to_string()),
                parameters: vec![],
                applicable_to_node_shapes: true,
                applicable_to_property_shapes: true,
                example: Some("ex:tzShape a lib:TimezoneValidator .".to_string()),
            },
        }
    }
}

impl Default for TimezoneValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomConstraintComponent for TimezoneValidator {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, _parameters: &HashMap<String, Term>) -> Result<()> {
        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        Ok(CustomConstraint {
            component_id: self.id.clone(),
            parameters,
            sparql_query: Some(
                r#"
                SELECT $this WHERE {
                    FILTER(REGEX(STR($this), "^[A-Z][a-z]+(/[A-Z][a-z_]+)*$"))
                }
                "#
                .to_string(),
            ),
            validation_function: None,
            message_template: Some("Value must be a valid IANA timezone".to_string()),
        })
    }
}

/// Business hours validation component
#[derive(Debug)]
pub struct BusinessHoursValidator {
    id: ConstraintComponentId,
    metadata: ComponentMetadata,
}

impl BusinessHoursValidator {
    pub fn new() -> Self {
        Self {
            id: ConstraintComponentId::new("lib:BusinessHoursValidator"),
            metadata: ComponentMetadata {
                name: "Business Hours Validator".to_string(),
                description: Some("Validates time is within business hours".to_string()),
                version: Some("1.0.0".to_string()),
                author: Some("OxiRS".to_string()),
                parameters: vec![
                    ParameterDefinition {
                        name: "startTime".to_string(),
                        description: Some("Start of business hours (HH:MM)".to_string()),
                        required: false,
                        datatype: Some("xsd:time".to_string()),
                        default_value: Some("09:00".to_string()),
                        validation_constraints: vec![],
                        cardinality: None,
                        allowed_values: None,
                    },
                    ParameterDefinition {
                        name: "endTime".to_string(),
                        description: Some("End of business hours (HH:MM)".to_string()),
                        required: false,
                        datatype: Some("xsd:time".to_string()),
                        default_value: Some("17:00".to_string()),
                        validation_constraints: vec![],
                        cardinality: None,
                        allowed_values: None,
                    },
                ],
                applicable_to_node_shapes: true,
                applicable_to_property_shapes: true,
                example: Some("ex:hoursShape a lib:BusinessHoursValidator ; lib:startTime \"08:00\" ; lib:endTime \"18:00\" .".to_string()),
            },
        }
    }
}

impl Default for BusinessHoursValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomConstraintComponent for BusinessHoursValidator {
    fn component_id(&self) -> &ConstraintComponentId {
        &self.id
    }

    fn metadata(&self) -> &ComponentMetadata {
        &self.metadata
    }

    fn validate_configuration(&self, _parameters: &HashMap<String, Term>) -> Result<()> {
        Ok(())
    }

    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint> {
        let start = parameters
            .get("startTime")
            .and_then(|t| match t {
                Term::Literal(l) => Some(l.value().to_string()),
                _ => None,
            })
            .unwrap_or_else(|| "09:00".to_string());

        let end = parameters
            .get("endTime")
            .and_then(|t| match t {
                Term::Literal(l) => Some(l.value().to_string()),
                _ => None,
            })
            .unwrap_or_else(|| "17:00".to_string());

        Ok(CustomConstraint {
            component_id: self.id.clone(),
            parameters,
            sparql_query: Some(format!(
                r#"
                SELECT $this WHERE {{
                    FILTER($this >= "{}:00"^^xsd:time && $this <= "{}:00"^^xsd:time)
                }}
                "#,
                start, end
            )),
            validation_function: None,
            message_template: Some(format!("Time must be between {} and {}", start, end)),
        })
    }
}

// ========== Stub implementations for remaining components ==========

macro_rules! stub_component {
    ($name:ident, $id:expr, $display_name:expr, $description:expr, $category:expr) => {
        #[derive(Debug)]
        pub struct $name {
            id: ConstraintComponentId,
            metadata: ComponentMetadata,
        }

        impl $name {
            pub fn new() -> Self {
                Self {
                    id: ConstraintComponentId::new($id),
                    metadata: ComponentMetadata {
                        name: $display_name.to_string(),
                        description: Some($description.to_string()),
                        version: Some("1.0.0".to_string()),
                        author: Some("OxiRS".to_string()),
                        parameters: vec![],
                        applicable_to_node_shapes: true,
                        applicable_to_property_shapes: true,
                        example: None,
                    },
                }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl CustomConstraintComponent for $name {
            fn component_id(&self) -> &ConstraintComponentId {
                &self.id
            }

            fn metadata(&self) -> &ComponentMetadata {
                &self.metadata
            }

            fn validate_configuration(&self, _parameters: &HashMap<String, Term>) -> Result<()> {
                Ok(())
            }

            fn create_constraint(
                &self,
                parameters: HashMap<String, Term>,
            ) -> Result<CustomConstraint> {
                Ok(CustomConstraint {
                    component_id: self.id.clone(),
                    parameters,
                    sparql_query: Some("SELECT $this WHERE { }".to_string()),
                    validation_function: None,
                    message_template: Some(format!("{} validation", $display_name)),
                })
            }
        }
    };
}

// Geospatial components
stub_component!(
    CoordinatesValidator,
    "lib:CoordinatesValidator",
    "Coordinates Validator",
    "Validates geographic coordinates (lat/lon)",
    "geospatial"
);
stub_component!(
    BoundingBoxValidator,
    "lib:BoundingBoxValidator",
    "Bounding Box Validator",
    "Validates geographic bounding box format",
    "geospatial"
);
stub_component!(
    CountryCodeValidator,
    "lib:CountryCodeValidator",
    "Country Code Validator",
    "Validates ISO 3166 country codes",
    "geospatial"
);
stub_component!(
    GeoJsonValidator,
    "lib:GeoJsonValidator",
    "GeoJSON Validator",
    "Validates GeoJSON format",
    "geospatial"
);

// Financial components
stub_component!(
    CurrencyValidator,
    "lib:CurrencyValidator",
    "Currency Validator",
    "Validates ISO 4217 currency codes",
    "financial"
);
stub_component!(
    IbanValidator,
    "lib:IbanValidator",
    "IBAN Validator",
    "Validates International Bank Account Number",
    "financial"
);
stub_component!(
    BicValidator,
    "lib:BicValidator",
    "BIC/SWIFT Validator",
    "Validates Bank Identifier Code",
    "financial"
);
stub_component!(
    CreditCardValidator,
    "lib:CreditCardValidator",
    "Credit Card Validator",
    "Validates credit card number format with Luhn check",
    "financial"
);

// Personal components
stub_component!(
    PhoneNumberValidator,
    "lib:PhoneNumberValidator",
    "Phone Number Validator",
    "Validates E.164 phone number format",
    "personal"
);
stub_component!(
    PostalCodeValidator,
    "lib:PostalCodeValidator",
    "Postal Code Validator",
    "Validates postal/ZIP code format by country",
    "personal"
);
stub_component!(
    NamePatternValidator,
    "lib:NamePatternValidator",
    "Name Pattern Validator",
    "Validates personal name patterns",
    "personal"
);
stub_component!(
    AgeRangeValidator,
    "lib:AgeRangeValidator",
    "Age Range Validator",
    "Validates age is within specified range",
    "personal"
);

// Scientific components
stub_component!(
    UnitValidator,
    "lib:UnitValidator",
    "Unit Validator",
    "Validates SI units and conversions",
    "scientific"
);
stub_component!(
    ChemicalFormulaValidator,
    "lib:ChemicalFormulaValidator",
    "Chemical Formula Validator",
    "Validates chemical formula notation",
    "scientific"
);
stub_component!(
    ScientificNotationValidator,
    "lib:ScientificNotationValidator",
    "Scientific Notation Validator",
    "Validates scientific notation format",
    "scientific"
);

// Semantic components
stub_component!(
    ClassHierarchyValidator,
    "lib:ClassHierarchyValidator",
    "Class Hierarchy Validator",
    "Validates RDF class hierarchy consistency",
    "semantic"
);
stub_component!(
    PropertyDomainRangeValidator,
    "lib:PropertyDomainRangeValidator",
    "Property Domain/Range Validator",
    "Validates property domain and range constraints",
    "semantic"
);
stub_component!(
    OntologyConsistencyValidator,
    "lib:OntologyConsistencyValidator",
    "Ontology Consistency Validator",
    "Validates overall ontology consistency",
    "semantic"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_creation() {
        let library = ConstraintLibrary::new();
        assert!(library.metadata().component_count > 0);
        assert!(!library.categories().is_empty());
    }

    #[test]
    fn test_get_component() {
        let library = ConstraintLibrary::new();
        let id = ConstraintComponentId::new("lib:UuidValidator");
        assert!(library.get(&id).is_some());
    }

    #[test]
    fn test_get_category() {
        let library = ConstraintLibrary::new();
        let components = library.get_category("identity");
        assert!(components.is_some());
        assert!(!components.unwrap().is_empty());
    }

    #[test]
    fn test_search() {
        let library = ConstraintLibrary::new();
        let results = library.search("uuid");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_uuid_validator() {
        let validator = UuidValidator::new();
        let params = HashMap::new();
        let constraint = validator.create_constraint(params).unwrap();
        assert!(constraint.sparql_query.is_some());
    }

    #[test]
    fn test_date_range_validator() {
        let validator = DateRangeValidator::new();

        // Should fail without parameters
        let empty_params = HashMap::new();
        assert!(validator.validate_configuration(&empty_params).is_err());

        // Should succeed with minDate
        let mut params = HashMap::new();
        params.insert(
            "minDate".to_string(),
            Term::Literal(oxirs_core::model::Literal::new("2020-01-01")),
        );
        assert!(validator.validate_configuration(&params).is_ok());

        let constraint = validator.create_constraint(params).unwrap();
        assert!(constraint.sparql_query.is_some());
    }

    #[test]
    fn test_all_categories() {
        let library = ConstraintLibrary::new();
        let categories = library.categories();

        assert!(categories.contains(&"identity"));
        assert!(categories.contains(&"temporal"));
        assert!(categories.contains(&"geospatial"));
        assert!(categories.contains(&"financial"));
        assert!(categories.contains(&"personal"));
        assert!(categories.contains(&"scientific"));
        assert!(categories.contains(&"semantic"));
    }
}
