//! Design Wizard
//!
//! Provides step-by-step guided shape creation with contextual help
//! and intelligent defaults based on domain and property names.

use super::{ConstraintSpec, DesignStep, Domain, PropertyDesign, PropertyHint, ShapeDesign};
use crate::{Result, Severity, ShaclError, Shape};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Design wizard for guided shape creation
#[derive(Debug)]
pub struct DesignWizard {
    /// Current design
    design: ShapeDesign,
    /// Wizard configuration
    config: WizardConfig,
    /// Step-specific state
    step_state: HashMap<u8, StepState>,
    /// Validation errors blocking progress
    blocking_errors: Vec<String>,
}

/// Wizard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WizardConfig {
    /// Allow skipping optional steps
    pub allow_skip: bool,
    /// Auto-suggest constraints based on property names
    pub auto_suggest: bool,
    /// Validate on each step transition
    pub validate_on_step: bool,
    /// Default domain
    pub default_domain: Domain,
    /// Default severity
    pub default_severity: Severity,
    /// Namespace prefixes for autocomplete
    pub namespace_prefixes: HashMap<String, String>,
}

impl Default for WizardConfig {
    fn default() -> Self {
        let mut prefixes = HashMap::new();
        prefixes.insert("ex".to_string(), "http://example.org/".to_string());
        prefixes.insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());
        prefixes.insert("schema".to_string(), "http://schema.org/".to_string());
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert("sh".to_string(), "http://www.w3.org/ns/shacl#".to_string());
        prefixes.insert("dcat".to_string(), "http://www.w3.org/ns/dcat#".to_string());
        prefixes.insert(
            "dc".to_string(),
            "http://purl.org/dc/elements/1.1/".to_string(),
        );
        prefixes.insert(
            "dcterms".to_string(),
            "http://purl.org/dc/terms/".to_string(),
        );

        Self {
            allow_skip: true,
            auto_suggest: true,
            validate_on_step: true,
            default_domain: Domain::Custom,
            default_severity: Severity::Violation,
            namespace_prefixes: prefixes,
        }
    }
}

/// State for a specific step
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepState {
    /// Whether step is completed
    pub completed: bool,
    /// User inputs for this step
    pub inputs: HashMap<String, String>,
    /// Validation messages
    pub messages: Vec<StepMessage>,
}

/// Message for a step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMessage {
    /// Message level
    pub level: MessageLevel,
    /// Message text
    pub text: String,
}

/// Message level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageLevel {
    Info,
    Warning,
    Error,
    Success,
}

/// Wizard step guidance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepGuidance {
    /// Step title
    pub title: String,
    /// Step description
    pub description: String,
    /// Required fields
    pub required_fields: Vec<FieldInfo>,
    /// Optional fields
    pub optional_fields: Vec<FieldInfo>,
    /// Tips and suggestions
    pub tips: Vec<String>,
    /// Examples
    pub examples: Vec<Example>,
}

/// Field information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldInfo {
    /// Field name
    pub name: String,
    /// Field label
    pub label: String,
    /// Field description
    pub description: String,
    /// Field type
    pub field_type: FieldType,
    /// Placeholder text
    pub placeholder: Option<String>,
    /// Validation pattern
    pub pattern: Option<String>,
    /// Default value
    pub default_value: Option<String>,
}

/// Field type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    Text,
    TextArea,
    Select(Vec<SelectOption>),
    MultiSelect(Vec<SelectOption>),
    Checkbox,
    Radio(Vec<SelectOption>),
    Number,
    IRI,
}

/// Select option
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectOption {
    pub value: String,
    pub label: String,
    pub description: Option<String>,
}

/// Example for a step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Example title
    pub title: String,
    /// Example description
    pub description: String,
    /// Example code/values
    pub code: String,
}

impl DesignWizard {
    /// Create a new wizard
    pub fn new(shape_id: impl Into<String>) -> Self {
        Self {
            design: ShapeDesign::new(shape_id),
            config: WizardConfig::default(),
            step_state: HashMap::new(),
            blocking_errors: Vec::new(),
        }
    }

    /// Create wizard with configuration
    pub fn with_config(mut self, config: WizardConfig) -> Self {
        self.config = config;
        self.design.domain = self.config.default_domain;
        self.design.severity = self.config.default_severity;
        self
    }

    /// Set domain for the design
    pub fn with_domain(mut self, domain: Domain) -> Self {
        self.design.domain = domain;
        self
    }

    /// Get current step
    pub fn current_step(&self) -> &DesignStep {
        &self.design.current_step
    }

    /// Get step guidance for current step
    pub fn get_guidance(&self) -> StepGuidance {
        self.get_step_guidance(&self.design.current_step)
    }

    /// Get guidance for a specific step
    pub fn get_step_guidance(&self, step: &DesignStep) -> StepGuidance {
        match step {
            DesignStep::BasicInfo => self.basic_info_guidance(),
            DesignStep::TargetDefinition => self.target_definition_guidance(),
            DesignStep::PropertyDefinition => self.property_definition_guidance(),
            DesignStep::Review => self.review_guidance(),
            DesignStep::Export => self.export_guidance(),
        }
    }

    fn basic_info_guidance(&self) -> StepGuidance {
        StepGuidance {
            title: "Basic Information".to_string(),
            description: "Define the basic information for your SHACL shape.".to_string(),
            required_fields: vec![FieldInfo {
                name: "id".to_string(),
                label: "Shape ID".to_string(),
                description: "Unique identifier (IRI) for this shape".to_string(),
                field_type: FieldType::IRI,
                placeholder: Some("ex:MyShape".to_string()),
                pattern: Some(r"^[a-zA-Z][a-zA-Z0-9_:-]*$".to_string()),
                default_value: None,
            }],
            optional_fields: vec![
                FieldInfo {
                    name: "label".to_string(),
                    label: "Label".to_string(),
                    description: "Human-readable label for the shape".to_string(),
                    field_type: FieldType::Text,
                    placeholder: Some("My Shape".to_string()),
                    pattern: None,
                    default_value: None,
                },
                FieldInfo {
                    name: "description".to_string(),
                    label: "Description".to_string(),
                    description: "Detailed description of what this shape validates".to_string(),
                    field_type: FieldType::TextArea,
                    placeholder: Some("Validates...".to_string()),
                    pattern: None,
                    default_value: None,
                },
                FieldInfo {
                    name: "domain".to_string(),
                    label: "Domain".to_string(),
                    description: "Domain category for templates and suggestions".to_string(),
                    field_type: FieldType::Select(
                        Domain::all()
                            .iter()
                            .map(|d| SelectOption {
                                value: d.name().to_string(),
                                label: d.name().to_string(),
                                description: Some(d.description().to_string()),
                            })
                            .collect(),
                    ),
                    placeholder: None,
                    pattern: None,
                    default_value: Some("Custom".to_string()),
                },
                FieldInfo {
                    name: "severity".to_string(),
                    label: "Default Severity".to_string(),
                    description: "Default severity for constraint violations".to_string(),
                    field_type: FieldType::Select(vec![
                        SelectOption {
                            value: "Violation".to_string(),
                            label: "Violation".to_string(),
                            description: Some("Error - data does not conform".to_string()),
                        },
                        SelectOption {
                            value: "Warning".to_string(),
                            label: "Warning".to_string(),
                            description: Some("Warning - potential issue".to_string()),
                        },
                        SelectOption {
                            value: "Info".to_string(),
                            label: "Info".to_string(),
                            description: Some("Informational only".to_string()),
                        },
                    ]),
                    placeholder: None,
                    pattern: None,
                    default_value: Some("Violation".to_string()),
                },
            ],
            tips: vec![
                "Use meaningful, descriptive shape IDs".to_string(),
                "Select a domain to get relevant template suggestions".to_string(),
                "Labels and descriptions help with documentation".to_string(),
            ],
            examples: vec![Example {
                title: "Person Shape".to_string(),
                description: "Shape for validating person data".to_string(),
                code: "ID: ex:PersonShape\nLabel: Person Shape\nDomain: Identity".to_string(),
            }],
        }
    }

    fn target_definition_guidance(&self) -> StepGuidance {
        StepGuidance {
            title: "Target Definition".to_string(),
            description: "Define which nodes this shape applies to.".to_string(),
            required_fields: vec![],
            optional_fields: vec![
                FieldInfo {
                    name: "target_class".to_string(),
                    label: "Target Class".to_string(),
                    description: "Target nodes that are instances of this class".to_string(),
                    field_type: FieldType::IRI,
                    placeholder: Some("foaf:Person".to_string()),
                    pattern: None,
                    default_value: None,
                },
                FieldInfo {
                    name: "target_node".to_string(),
                    label: "Target Node".to_string(),
                    description: "Target a specific node by IRI".to_string(),
                    field_type: FieldType::IRI,
                    placeholder: Some("ex:john".to_string()),
                    pattern: None,
                    default_value: None,
                },
                FieldInfo {
                    name: "target_subjects_of".to_string(),
                    label: "Subjects Of".to_string(),
                    description: "Target subjects of a property".to_string(),
                    field_type: FieldType::IRI,
                    placeholder: Some("foaf:knows".to_string()),
                    pattern: None,
                    default_value: None,
                },
                FieldInfo {
                    name: "target_objects_of".to_string(),
                    label: "Objects Of".to_string(),
                    description: "Target objects of a property".to_string(),
                    field_type: FieldType::IRI,
                    placeholder: Some("foaf:knows".to_string()),
                    pattern: None,
                    default_value: None,
                },
            ],
            tips: vec![
                "At least one target should be defined".to_string(),
                "sh:targetClass is the most common way to select nodes".to_string(),
                "Multiple targets create a union of selected nodes".to_string(),
            ],
            examples: vec![
                Example {
                    title: "Target by Class".to_string(),
                    description: "Select all instances of foaf:Person".to_string(),
                    code: "Target Class: foaf:Person".to_string(),
                },
                Example {
                    title: "Target by Relationship".to_string(),
                    description: "Select subjects of foaf:knows".to_string(),
                    code: "Subjects Of: foaf:knows".to_string(),
                },
            ],
        }
    }

    fn property_definition_guidance(&self) -> StepGuidance {
        StepGuidance {
            title: "Properties & Constraints".to_string(),
            description: "Define property constraints for the shape.".to_string(),
            required_fields: vec![],
            optional_fields: vec![FieldInfo {
                name: "property_path".to_string(),
                label: "Property Path".to_string(),
                description: "Property to constrain".to_string(),
                field_type: FieldType::IRI,
                placeholder: Some("foaf:name".to_string()),
                pattern: None,
                default_value: None,
            }],
            tips: vec![
                "Use property hints to quickly add common constraints".to_string(),
                "Required properties should have minCount 1".to_string(),
                "Consider datatype constraints for type safety".to_string(),
                "Pattern constraints validate string formats".to_string(),
            ],
            examples: vec![
                Example {
                    title: "Required String Property".to_string(),
                    description: "Name must be a non-empty string".to_string(),
                    code: "Path: foaf:name\nminCount: 1\ndatatype: xsd:string".to_string(),
                },
                Example {
                    title: "Email Property".to_string(),
                    description: "Email with pattern validation".to_string(),
                    code: "Path: foaf:mbox\npattern: ^[^@]+@[^@]+$".to_string(),
                },
            ],
        }
    }

    fn review_guidance(&self) -> StepGuidance {
        let issues = self.design.issues.clone();

        StepGuidance {
            title: "Review & Optimize".to_string(),
            description: "Review your shape and address any issues.".to_string(),
            required_fields: vec![],
            optional_fields: vec![],
            tips: issues
                .iter()
                .map(|i| format!("[{:?}] {}", i.severity, i.message))
                .collect(),
            examples: vec![],
        }
    }

    fn export_guidance(&self) -> StepGuidance {
        StepGuidance {
            title: "Export".to_string(),
            description: "Export your shape in various formats.".to_string(),
            required_fields: vec![],
            optional_fields: vec![FieldInfo {
                name: "format".to_string(),
                label: "Export Format".to_string(),
                description: "Choose the output format".to_string(),
                field_type: FieldType::Select(vec![
                    SelectOption {
                        value: "turtle".to_string(),
                        label: "Turtle".to_string(),
                        description: Some("RDF Turtle format".to_string()),
                    },
                    SelectOption {
                        value: "json-ld".to_string(),
                        label: "JSON-LD".to_string(),
                        description: Some("JSON-LD format".to_string()),
                    },
                    SelectOption {
                        value: "rdf-xml".to_string(),
                        label: "RDF/XML".to_string(),
                        description: Some("RDF/XML format".to_string()),
                    },
                ]),
                placeholder: None,
                pattern: None,
                default_value: Some("turtle".to_string()),
            }],
            tips: vec![
                "Turtle is the most human-readable format".to_string(),
                "JSON-LD is useful for web applications".to_string(),
            ],
            examples: vec![],
        }
    }

    /// Set field value for current step
    pub fn set_field(&mut self, name: &str, value: impl Into<String>) {
        let step_num = self.design.current_step.number();
        let state = self.step_state.entry(step_num).or_default();
        state.inputs.insert(name.to_string(), value.into());

        // Apply to design based on field name
        match name {
            "label" => self.design.label = Some(state.inputs[name].clone()),
            "description" => self.design.description = Some(state.inputs[name].clone()),
            _ => {}
        }
    }

    /// Get field value for current step
    pub fn get_field(&self, name: &str) -> Option<&String> {
        let step_num = self.design.current_step.number();
        self.step_state
            .get(&step_num)
            .and_then(|s| s.inputs.get(name))
    }

    /// Add target class
    pub fn add_target_class(&mut self, class: impl Into<String>) {
        self.design.add_target_class(class);
    }

    /// Add target node
    pub fn add_target_node(&mut self, node: impl Into<String>) {
        self.design.add_target_node(node);
    }

    /// Add property with hints
    pub fn add_property(&mut self, path: impl Into<String>) -> PropertyBuilder<'_> {
        PropertyBuilder {
            wizard: self,
            property: PropertyDesign::new(path),
        }
    }

    /// Move to next step
    pub fn advance(&mut self) -> Result<Option<DesignStep>> {
        if self.config.validate_on_step {
            self.validate_current_step()?;
        }

        let step_num = self.design.current_step.number();
        if let Some(state) = self.step_state.get_mut(&step_num) {
            state.completed = true;
        }

        Ok(self.design.next_step())
    }

    /// Move to previous step
    pub fn back(&mut self) -> Option<DesignStep> {
        self.design.prev_step()
    }

    /// Skip current step (if allowed)
    pub fn skip(&mut self) -> Result<Option<DesignStep>> {
        if !self.config.allow_skip {
            return Err(ShaclError::Configuration(
                "Skipping is not allowed".to_string(),
            ));
        }
        Ok(self.design.next_step())
    }

    /// Validate current step
    pub fn validate_current_step(&mut self) -> Result<()> {
        self.blocking_errors.clear();

        match self.design.current_step {
            DesignStep::BasicInfo => {
                if self.design.id.is_empty() {
                    self.blocking_errors
                        .push("Shape ID is required".to_string());
                }
            }
            DesignStep::TargetDefinition => {
                // Warning but not blocking
            }
            DesignStep::PropertyDefinition => {
                // Warning but not blocking
            }
            DesignStep::Review => {
                self.design.validate();
                for issue in &self.design.issues {
                    if matches!(issue.severity, super::IssueSeverity::Error) {
                        self.blocking_errors.push(issue.message.clone());
                    }
                }
            }
            DesignStep::Export => {}
        }

        if self.blocking_errors.is_empty() {
            Ok(())
        } else {
            Err(ShaclError::Configuration(self.blocking_errors.join("; ")))
        }
    }

    /// Get blocking errors
    pub fn blocking_errors(&self) -> &[String] {
        &self.blocking_errors
    }

    /// Get the design
    pub fn design(&self) -> &ShapeDesign {
        &self.design
    }

    /// Get mutable design
    pub fn design_mut(&mut self) -> &mut ShapeDesign {
        &mut self.design
    }

    /// Build the shape
    pub fn build(self) -> Result<Shape> {
        self.design.build()
    }

    /// Get completion percentage
    pub fn completion(&self) -> u8 {
        self.design.completion_percentage()
    }

    /// Expand prefixed IRI
    pub fn expand_prefix(&self, prefixed: &str) -> Option<String> {
        if let Some(colon_pos) = prefixed.find(':') {
            let prefix = &prefixed[..colon_pos];
            let local = &prefixed[colon_pos + 1..];
            self.config
                .namespace_prefixes
                .get(prefix)
                .map(|ns| format!("{}{}", ns, local))
        } else {
            None
        }
    }

    /// Compact IRI to prefixed form
    pub fn compact_iri(&self, iri: &str) -> String {
        for (prefix, namespace) in &self.config.namespace_prefixes {
            if iri.starts_with(namespace) {
                return format!("{}:{}", prefix, &iri[namespace.len()..]);
            }
        }
        iri.to_string()
    }
}

/// Builder for adding properties
pub struct PropertyBuilder<'a> {
    wizard: &'a mut DesignWizard,
    property: PropertyDesign,
}

impl<'a> PropertyBuilder<'a> {
    /// Add hint
    pub fn with_hint(mut self, hint: PropertyHint) -> Self {
        self.property.hints.insert(hint);
        self
    }

    /// Set required
    pub fn required(self) -> Self {
        self.with_hint(PropertyHint::Required)
    }

    /// Set unique
    pub fn unique(self) -> Self {
        self.with_hint(PropertyHint::Unique)
    }

    /// Set string datatype
    pub fn string(self) -> Self {
        self.with_hint(PropertyHint::String)
    }

    /// Set integer datatype
    pub fn integer(self) -> Self {
        self.with_hint(PropertyHint::Integer)
    }

    /// Set date datatype
    pub fn date(self) -> Self {
        self.with_hint(PropertyHint::Date)
    }

    /// Set email pattern
    pub fn email(self) -> Self {
        self.with_hint(PropertyHint::Email)
    }

    /// Add custom constraint
    pub fn with_constraint(mut self, constraint: ConstraintSpec) -> Self {
        self.property.constraints.push(constraint);
        self
    }

    /// Set min count
    pub fn min_count(self, count: u32) -> Self {
        self.with_constraint(ConstraintSpec::MinCount(count))
    }

    /// Set max count
    pub fn max_count(self, count: u32) -> Self {
        self.with_constraint(ConstraintSpec::MaxCount(count))
    }

    /// Set pattern
    pub fn pattern(self, pattern: impl Into<String>) -> Self {
        self.with_constraint(ConstraintSpec::Pattern(pattern.into()))
    }

    /// Set label
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.property.label = Some(label.into());
        self
    }

    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.property.description = Some(desc.into());
        self
    }

    /// Reference another shape
    pub fn referencing(mut self, shape_id: impl Into<String>) -> Self {
        self.property.shape_ref = Some(shape_id.into());
        self.property.hints.insert(PropertyHint::Reference);
        self
    }

    /// Done - add property to wizard
    pub fn done(self) {
        self.wizard.design.add_property(self.property);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wizard_creation() {
        let wizard = DesignWizard::new("ex:TestShape").with_domain(Domain::Identity);

        assert_eq!(wizard.design().id, "ex:TestShape");
        assert_eq!(wizard.design().domain, Domain::Identity);
    }

    #[test]
    fn test_wizard_steps() {
        let mut wizard = DesignWizard::new("ex:Test");

        assert_eq!(wizard.current_step().number(), 1);

        wizard.advance().ok();
        assert_eq!(wizard.current_step().number(), 2);

        wizard.back();
        assert_eq!(wizard.current_step().number(), 1);
    }

    #[test]
    fn test_add_property() {
        let mut wizard = DesignWizard::new("ex:Test");

        wizard
            .add_property("foaf:name")
            .required()
            .string()
            .label("Name")
            .done();

        assert_eq!(wizard.design().properties.len(), 1);
        let prop = &wizard.design().properties[0];
        assert!(prop.is_required());
        assert!(prop.hints.contains(&PropertyHint::String));
    }

    #[test]
    fn test_guidance() {
        let wizard = DesignWizard::new("ex:Test");
        let guidance = wizard.get_guidance();

        assert!(!guidance.title.is_empty());
        assert!(!guidance.description.is_empty());
    }

    #[test]
    fn test_prefix_expansion() {
        let wizard = DesignWizard::new("ex:Test");

        let expanded = wizard.expand_prefix("foaf:Person");
        assert_eq!(
            expanded,
            Some("http://xmlns.com/foaf/0.1/Person".to_string())
        );

        let compacted = wizard.compact_iri("http://xmlns.com/foaf/0.1/name");
        assert_eq!(compacted, "foaf:name");
    }

    #[test]
    fn test_build() {
        let mut wizard = DesignWizard::new("ex:PersonShape");
        wizard.add_target_class("foaf:Person");
        wizard.add_property("foaf:name").required().string().done();

        let shape = wizard.build().expect("operation should succeed");
        assert!(!shape.targets.is_empty());
    }
}
