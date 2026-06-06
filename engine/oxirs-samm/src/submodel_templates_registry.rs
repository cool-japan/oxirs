//! Submodel Templates — Registry
//!
//! Template registry: register/lookup/list templates and version management.

use std::collections::HashMap;

use super::submodel_templates_engine::check_constraint;
use super::submodel_templates_types::{
    ElementType, Multiplicity, SubmodelTemplate, TemplateCategory, TemplateConstraint,
    TemplateElement, TemplateValidationError, TemplateValidationResult, TemplateVersion,
    ValidationSeverity, ValueType,
};

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
        self.templates
            .entry(template.semantic_id.clone())
            .or_default()
            .push(template.clone());

        self.aliases
            .insert(template.idta_id.clone(), template.semantic_id.clone());

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

        for opt in &template.optional_elements {
            if elements.contains_key(&opt.id_short) {
                matched += 1;
            }
        }

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

        for constraint in &template.constraints {
            if let Some(err) = check_constraint(constraint, elements) {
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

    // ── Standard template definitions ─────────────────────────────────────────

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
