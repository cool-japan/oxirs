//! Interactive Shape Designer
//!
//! Provides guided, wizard-style tools for designing SHACL shapes interactively.
//! This module helps users create well-structured shapes through:
//!
//! - Step-by-step shape creation wizard
//! - Domain-specific templates and recommendations
//! - Data sampling and inference
//! - Shape validation and optimization hints
//! - Export to multiple formats
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_shacl::designer::{ShapeDesigner, DesignWizard};
//!
//! // Create a wizard for designing a Person shape
//! let wizard = DesignWizard::new("PersonShape")
//!     .with_domain(Domain::Identity)
//!     .start();
//!
//! // Interactive step-by-step design
//! wizard.add_target_class("foaf:Person");
//! wizard.add_property("foaf:name")
//!     .with_hint(PropertyHint::Required)
//!     .with_hint(PropertyHint::String);
//!
//! let shape = wizard.build()?;
//! ```

pub mod inference;
pub mod recommendations;
pub mod wizard;

pub use inference::*;
pub use recommendations::*;
pub use wizard::*;

use crate::{
    constraints::{
        cardinality_constraints::{MaxCountConstraint, MinCountConstraint},
        string_constraints::{MaxLengthConstraint, MinLengthConstraint, PatternConstraint},
        value_constraints::DatatypeConstraint,
        Constraint,
    },
    templates::{TemplateCategory, TemplateLibrary},
    ConstraintComponentId, Result, Severity, ShaclError, Shape, ShapeId, ShapeType, Target,
};
use oxirs_core::NamedNode;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Domain categories for shape design
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    /// Person and identity data (FOAF, Schema.org Person)
    Identity,
    /// Contact information (email, phone, address)
    Contact,
    /// E-commerce and products
    Commerce,
    /// Web resources and digital content
    Web,
    /// Temporal and scheduling data
    Temporal,
    /// Geographic and spatial data
    Geospatial,
    /// Financial and monetary data
    Financial,
    /// Scientific and research data
    Scientific,
    /// Healthcare and medical data
    Healthcare,
    /// Legal and compliance data
    Legal,
    /// Custom domain
    Custom,
}

impl Domain {
    /// Get the template category for this domain
    pub fn to_template_category(self) -> Option<TemplateCategory> {
        match self {
            Domain::Identity => Some(TemplateCategory::Identity),
            Domain::Contact => Some(TemplateCategory::Contact),
            Domain::Commerce => Some(TemplateCategory::Commerce),
            Domain::Web => Some(TemplateCategory::Web),
            Domain::Temporal => Some(TemplateCategory::Temporal),
            Domain::Geospatial => Some(TemplateCategory::Geospatial),
            Domain::Financial => Some(TemplateCategory::Financial),
            Domain::Scientific => Some(TemplateCategory::Scientific),
            _ => None,
        }
    }

    /// Get all available domains
    pub fn all() -> Vec<Domain> {
        vec![
            Domain::Identity,
            Domain::Contact,
            Domain::Commerce,
            Domain::Web,
            Domain::Temporal,
            Domain::Geospatial,
            Domain::Financial,
            Domain::Scientific,
            Domain::Healthcare,
            Domain::Legal,
            Domain::Custom,
        ]
    }

    /// Get domain name
    pub fn name(&self) -> &'static str {
        match self {
            Domain::Identity => "Identity",
            Domain::Contact => "Contact",
            Domain::Commerce => "Commerce",
            Domain::Web => "Web",
            Domain::Temporal => "Temporal",
            Domain::Geospatial => "Geospatial",
            Domain::Financial => "Financial",
            Domain::Scientific => "Scientific",
            Domain::Healthcare => "Healthcare",
            Domain::Legal => "Legal",
            Domain::Custom => "Custom",
        }
    }

    /// Get domain description
    pub fn description(&self) -> &'static str {
        match self {
            Domain::Identity => "Person, organization, and identity data",
            Domain::Contact => "Email, phone, address, and contact information",
            Domain::Commerce => "Products, orders, and e-commerce data",
            Domain::Web => "Web pages, URLs, and digital resources",
            Domain::Temporal => "Dates, times, events, and scheduling",
            Domain::Geospatial => "Locations, coordinates, and geographic data",
            Domain::Financial => "Currency, prices, and financial transactions",
            Domain::Scientific => "Research datasets, experiments, and measurements",
            Domain::Healthcare => "Medical records, patients, and health data",
            Domain::Legal => "Contracts, compliance, and legal documents",
            Domain::Custom => "Custom domain-specific shapes",
        }
    }
}

/// Design step in the wizard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DesignStep {
    /// Choose domain and basic info
    BasicInfo,
    /// Define target selection
    TargetDefinition,
    /// Add properties and constraints
    PropertyDefinition,
    /// Review and optimize
    Review,
    /// Export and finalize
    Export,
}

impl DesignStep {
    /// Get next step
    pub fn next(&self) -> Option<DesignStep> {
        match self {
            DesignStep::BasicInfo => Some(DesignStep::TargetDefinition),
            DesignStep::TargetDefinition => Some(DesignStep::PropertyDefinition),
            DesignStep::PropertyDefinition => Some(DesignStep::Review),
            DesignStep::Review => Some(DesignStep::Export),
            DesignStep::Export => None,
        }
    }

    /// Get previous step
    pub fn prev(&self) -> Option<DesignStep> {
        match self {
            DesignStep::BasicInfo => None,
            DesignStep::TargetDefinition => Some(DesignStep::BasicInfo),
            DesignStep::PropertyDefinition => Some(DesignStep::TargetDefinition),
            DesignStep::Review => Some(DesignStep::PropertyDefinition),
            DesignStep::Export => Some(DesignStep::Review),
        }
    }

    /// Get step number (1-based)
    pub fn number(&self) -> u8 {
        match self {
            DesignStep::BasicInfo => 1,
            DesignStep::TargetDefinition => 2,
            DesignStep::PropertyDefinition => 3,
            DesignStep::Review => 4,
            DesignStep::Export => 5,
        }
    }

    /// Get step title
    pub fn title(&self) -> &'static str {
        match self {
            DesignStep::BasicInfo => "Basic Information",
            DesignStep::TargetDefinition => "Target Definition",
            DesignStep::PropertyDefinition => "Properties & Constraints",
            DesignStep::Review => "Review & Optimize",
            DesignStep::Export => "Export",
        }
    }
}

/// Property hint for constraint recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyHint {
    /// Property is required (minCount 1)
    Required,
    /// Property should be unique (maxCount 1)
    Unique,
    /// Property is optional
    Optional,
    /// Property is a string
    String,
    /// Property is an integer
    Integer,
    /// Property is a decimal/float
    Decimal,
    /// Property is a date
    Date,
    /// Property is a datetime
    DateTime,
    /// Property is a boolean
    Boolean,
    /// Property is an IRI/URI
    IRI,
    /// Property is an email
    Email,
    /// Property is a phone number
    Phone,
    /// Property is a URL
    URL,
    /// Property references another shape
    Reference,
    /// Property is multi-valued
    MultiValued,
}

impl PropertyHint {
    /// Get XSD datatype for this hint
    pub fn datatype(&self) -> Option<&'static str> {
        match self {
            PropertyHint::String => Some("http://www.w3.org/2001/XMLSchema#string"),
            PropertyHint::Integer => Some("http://www.w3.org/2001/XMLSchema#integer"),
            PropertyHint::Decimal => Some("http://www.w3.org/2001/XMLSchema#decimal"),
            PropertyHint::Date => Some("http://www.w3.org/2001/XMLSchema#date"),
            PropertyHint::DateTime => Some("http://www.w3.org/2001/XMLSchema#dateTime"),
            PropertyHint::Boolean => Some("http://www.w3.org/2001/XMLSchema#boolean"),
            PropertyHint::IRI => Some("http://www.w3.org/2001/XMLSchema#anyURI"),
            _ => None,
        }
    }

    /// Get pattern for this hint
    pub fn pattern(&self) -> Option<&'static str> {
        match self {
            PropertyHint::Email => Some(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
            PropertyHint::Phone => Some(r"^\+?[1-9]\d{1,14}$"),
            PropertyHint::URL => Some(r"^https?://[^\s/$.?#].[^\s]*$"),
            _ => None,
        }
    }
}

/// Property definition in design
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyDesign {
    /// Property path/predicate
    pub path: String,
    /// Human-readable label
    pub label: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Applied hints
    pub hints: HashSet<PropertyHint>,
    /// Custom constraints
    pub constraints: Vec<ConstraintSpec>,
    /// Reference to another shape
    pub shape_ref: Option<String>,
}

impl PropertyDesign {
    /// Create a new property design
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            label: None,
            description: None,
            hints: HashSet::new(),
            constraints: Vec::new(),
            shape_ref: None,
        }
    }

    /// Add a hint
    pub fn with_hint(mut self, hint: PropertyHint) -> Self {
        self.hints.insert(hint);
        self
    }

    /// Set label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a constraint spec
    pub fn with_constraint(mut self, constraint: ConstraintSpec) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Set shape reference
    pub fn referencing(mut self, shape_id: impl Into<String>) -> Self {
        self.shape_ref = Some(shape_id.into());
        self.hints.insert(PropertyHint::Reference);
        self
    }

    /// Check if property is required
    pub fn is_required(&self) -> bool {
        self.hints.contains(&PropertyHint::Required)
    }

    /// Check if property is unique
    pub fn is_unique(&self) -> bool {
        self.hints.contains(&PropertyHint::Unique)
    }
}

/// Constraint specification for design
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintSpec {
    MinCount(u32),
    MaxCount(u32),
    Datatype(String),
    Pattern(String),
    MinLength(u32),
    MaxLength(u32),
    MinInclusive(f64),
    MaxInclusive(f64),
    MinExclusive(f64),
    MaxExclusive(f64),
    In(Vec<String>),
    Class(String),
    NodeKind(String),
    HasValue(String),
    Equals(String),
    Disjoint(String),
    LessThan(String),
    LessThanOrEquals(String),
    Shape(String),
}

impl ConstraintSpec {
    /// Convert to SHACL constraint
    pub fn to_constraint(&self) -> Result<(ConstraintComponentId, Constraint)> {
        match self {
            ConstraintSpec::MinCount(count) => Ok((
                ConstraintComponentId::new("sh:minCount"),
                Constraint::MinCount(MinCountConstraint { min_count: *count }),
            )),
            ConstraintSpec::MaxCount(count) => Ok((
                ConstraintComponentId::new("sh:maxCount"),
                Constraint::MaxCount(MaxCountConstraint { max_count: *count }),
            )),
            ConstraintSpec::Datatype(dt) => {
                let node = NamedNode::new(dt).map_err(|e| {
                    ShaclError::Configuration(format!("Invalid datatype IRI: {}", e))
                })?;
                Ok((
                    ConstraintComponentId::new("sh:datatype"),
                    Constraint::Datatype(DatatypeConstraint { datatype_iri: node }),
                ))
            }
            ConstraintSpec::Pattern(pat) => Ok((
                ConstraintComponentId::new("sh:pattern"),
                Constraint::Pattern(PatternConstraint {
                    pattern: pat.clone(),
                    flags: None,
                    message: None,
                }),
            )),
            ConstraintSpec::MinLength(len) => Ok((
                ConstraintComponentId::new("sh:minLength"),
                Constraint::MinLength(MinLengthConstraint { min_length: *len }),
            )),
            ConstraintSpec::MaxLength(len) => Ok((
                ConstraintComponentId::new("sh:maxLength"),
                Constraint::MaxLength(MaxLengthConstraint { max_length: *len }),
            )),
            _ => Err(ShaclError::Configuration(format!(
                "Constraint type not yet supported: {:?}",
                self
            ))),
        }
    }
}

/// Shape design state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeDesign {
    /// Shape identifier
    pub id: String,
    /// Human-readable label
    pub label: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Domain category
    pub domain: Domain,
    /// Current design step
    pub current_step: DesignStep,
    /// Target classes
    pub target_classes: Vec<String>,
    /// Target nodes
    pub target_nodes: Vec<String>,
    /// Target subjects of properties
    pub target_subjects_of: Vec<String>,
    /// Target objects of properties
    pub target_objects_of: Vec<String>,
    /// Property designs
    pub properties: Vec<PropertyDesign>,
    /// Severity level
    pub severity: Severity,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Design history for undo/redo
    pub history: Vec<DesignAction>,
    /// Issues found during review
    pub issues: Vec<DesignIssue>,
}

impl ShapeDesign {
    /// Create a new shape design
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: None,
            description: None,
            domain: Domain::Custom,
            current_step: DesignStep::BasicInfo,
            target_classes: Vec::new(),
            target_nodes: Vec::new(),
            target_subjects_of: Vec::new(),
            target_objects_of: Vec::new(),
            properties: Vec::new(),
            severity: Severity::Violation,
            metadata: HashMap::new(),
            history: Vec::new(),
            issues: Vec::new(),
        }
    }

    /// Set domain
    pub fn with_domain(mut self, domain: Domain) -> Self {
        self.domain = domain;
        self
    }

    /// Set label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add target class
    pub fn add_target_class(&mut self, class: impl Into<String>) {
        let class = class.into();
        self.history
            .push(DesignAction::AddTargetClass(class.clone()));
        self.target_classes.push(class);
    }

    /// Add target node
    pub fn add_target_node(&mut self, node: impl Into<String>) {
        let node = node.into();
        self.history.push(DesignAction::AddTargetNode(node.clone()));
        self.target_nodes.push(node);
    }

    /// Add property
    pub fn add_property(&mut self, property: PropertyDesign) {
        self.history
            .push(DesignAction::AddProperty(property.path.clone()));
        self.properties.push(property);
    }

    /// Remove property by path
    pub fn remove_property(&mut self, path: &str) -> bool {
        if let Some(idx) = self.properties.iter().position(|p| p.path == path) {
            self.history
                .push(DesignAction::RemoveProperty(path.to_string()));
            self.properties.remove(idx);
            true
        } else {
            false
        }
    }

    /// Get property by path
    pub fn get_property(&self, path: &str) -> Option<&PropertyDesign> {
        self.properties.iter().find(|p| p.path == path)
    }

    /// Get mutable property by path
    pub fn get_property_mut(&mut self, path: &str) -> Option<&mut PropertyDesign> {
        self.properties.iter_mut().find(|p| p.path == path)
    }

    /// Move to next step
    pub fn next_step(&mut self) -> Option<DesignStep> {
        if let Some(next) = self.current_step.next() {
            self.current_step = next.clone();
            Some(next)
        } else {
            None
        }
    }

    /// Move to previous step
    pub fn prev_step(&mut self) -> Option<DesignStep> {
        if let Some(prev) = self.current_step.prev() {
            self.current_step = prev.clone();
            Some(prev)
        } else {
            None
        }
    }

    /// Go to specific step
    pub fn go_to_step(&mut self, step: DesignStep) {
        self.current_step = step;
    }

    /// Build the final shape
    pub fn build(&self) -> Result<Shape> {
        let shape_id = ShapeId::new(&self.id);
        let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

        // Set metadata
        shape.label = self.label.clone();
        shape.description = self.description.clone();
        shape.severity = self.severity;

        // Add targets
        for class in &self.target_classes {
            if let Ok(node) = NamedNode::new(class) {
                shape.targets.push(Target::Class(node));
            }
        }

        for node in &self.target_nodes {
            if let Ok(named) = NamedNode::new(node) {
                shape.targets.push(Target::Node(named.into()));
            }
        }

        for prop in &self.target_subjects_of {
            if let Ok(node) = NamedNode::new(prop) {
                shape.targets.push(Target::SubjectsOf(node));
            }
        }

        for prop in &self.target_objects_of {
            if let Ok(node) = NamedNode::new(prop) {
                shape.targets.push(Target::ObjectsOf(node));
            }
        }

        Ok(shape)
    }

    /// Validate the design
    pub fn validate(&mut self) -> Vec<DesignIssue> {
        self.issues.clear();

        // Check for basic issues
        if self.target_classes.is_empty()
            && self.target_nodes.is_empty()
            && self.target_subjects_of.is_empty()
            && self.target_objects_of.is_empty()
        {
            self.issues.push(DesignIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::Target,
                message: "No targets defined - shape will not match any nodes".to_string(),
                suggestion: Some("Add at least one target class or target node".to_string()),
            });
        }

        if self.properties.is_empty() {
            self.issues.push(DesignIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::Property,
                message: "No properties defined".to_string(),
                suggestion: Some("Add property constraints to validate data".to_string()),
            });
        }

        // Check each property
        for prop in &self.properties {
            if prop.hints.is_empty() && prop.constraints.is_empty() {
                self.issues.push(DesignIssue {
                    severity: IssueSeverity::Info,
                    category: IssueCategory::Property,
                    message: format!("Property '{}' has no constraints", prop.path),
                    suggestion: Some("Add hints or constraints to validate values".to_string()),
                });
            }

            // Check for conflicting hints
            if prop.hints.contains(&PropertyHint::Required)
                && prop.hints.contains(&PropertyHint::Optional)
            {
                self.issues.push(DesignIssue {
                    severity: IssueSeverity::Error,
                    category: IssueCategory::Conflict,
                    message: format!("Property '{}' is both Required and Optional", prop.path),
                    suggestion: Some("Remove one of the conflicting hints".to_string()),
                });
            }
        }

        self.issues.clone()
    }

    /// Get completion percentage (0-100)
    pub fn completion_percentage(&self) -> u8 {
        let mut score = 0u8;

        // Basic info (20%)
        if !self.id.is_empty() {
            score += 10;
        }
        if self.label.is_some() {
            score += 5;
        }
        if self.description.is_some() {
            score += 5;
        }

        // Targets (30%)
        if !self.target_classes.is_empty() || !self.target_nodes.is_empty() {
            score += 30;
        }

        // Properties (40%)
        if !self.properties.is_empty() {
            let prop_score = (self.properties.len().min(4) * 10) as u8;
            score += prop_score;
        }

        // Validation (10%)
        let issues = self
            .issues
            .iter()
            .filter(|i| matches!(i.severity, IssueSeverity::Error))
            .count();
        if issues == 0 {
            score += 10;
        }

        score.min(100)
    }
}

/// Design action for history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DesignAction {
    AddTargetClass(String),
    RemoveTargetClass(String),
    AddTargetNode(String),
    RemoveTargetNode(String),
    AddProperty(String),
    RemoveProperty(String),
    ModifyProperty(String),
    SetLabel(Option<String>),
    SetDescription(Option<String>),
    SetDomain(Domain),
}

/// Design issue found during validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue category
    pub category: IssueCategory,
    /// Issue message
    pub message: String,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Issue severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

/// Issue category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueCategory {
    Target,
    Property,
    Constraint,
    Conflict,
    Performance,
    BestPractice,
}

/// Shape designer - main entry point
#[derive(Debug)]
pub struct ShapeDesigner {
    /// Current designs
    designs: HashMap<String, ShapeDesign>,
    /// Template library
    templates: TemplateLibrary,
    /// Recommendations engine
    recommendations: RecommendationEngine,
}

impl ShapeDesigner {
    /// Create a new shape designer
    pub fn new() -> Self {
        Self {
            designs: HashMap::new(),
            templates: TemplateLibrary::new(),
            recommendations: RecommendationEngine::new(),
        }
    }

    /// Start a new design
    pub fn new_design(&mut self, id: impl Into<String>) -> &mut ShapeDesign {
        let id = id.into();
        self.designs
            .entry(id.clone())
            .or_insert_with(|| ShapeDesign::new(&id))
    }

    /// Get design by ID
    pub fn get_design(&self, id: &str) -> Option<&ShapeDesign> {
        self.designs.get(id)
    }

    /// Get mutable design by ID
    pub fn get_design_mut(&mut self, id: &str) -> Option<&mut ShapeDesign> {
        self.designs.get_mut(id)
    }

    /// List all design IDs
    pub fn list_designs(&self) -> Vec<String> {
        self.designs.keys().cloned().collect()
    }

    /// Remove a design
    pub fn remove_design(&mut self, id: &str) -> Option<ShapeDesign> {
        self.designs.remove(id)
    }

    /// Get templates by domain
    pub fn templates_for_domain(&self, domain: Domain) -> Vec<&crate::templates::ShapeTemplate> {
        if let Some(category) = domain.to_template_category() {
            self.templates.by_category(category)
        } else {
            Vec::new()
        }
    }

    /// Get recommendations for a property name
    pub fn recommend_for_property(&self, property_name: &str) -> Vec<PropertyHint> {
        self.recommendations.recommend_for_property(property_name)
    }

    /// Build all designs into shapes
    pub fn build_all(&self) -> Result<Vec<Shape>> {
        self.designs.values().map(|d| d.build()).collect()
    }

    /// Export design to JSON
    pub fn export_design_json(&self, id: &str) -> Result<String> {
        let design = self
            .designs
            .get(id)
            .ok_or_else(|| ShaclError::Configuration(format!("Design '{}' not found", id)))?;

        serde_json::to_string_pretty(design).map_err(|e| ShaclError::Json(e.to_string()))
    }

    /// Import design from JSON
    pub fn import_design_json(&mut self, json: &str) -> Result<String> {
        let design: ShapeDesign =
            serde_json::from_str(json).map_err(|e| ShaclError::Json(e.to_string()))?;
        let id = design.id.clone();
        self.designs.insert(id.clone(), design);
        Ok(id)
    }
}

impl Default for ShapeDesigner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_design_creation() {
        let design = ShapeDesign::new("ex:PersonShape")
            .with_domain(Domain::Identity)
            .with_label("Person Shape")
            .with_description("Validates person data");

        assert_eq!(design.id, "ex:PersonShape");
        assert_eq!(design.domain, Domain::Identity);
        assert_eq!(design.label, Some("Person Shape".to_string()));
    }

    #[test]
    fn test_property_design() {
        let prop = PropertyDesign::new("foaf:name")
            .with_label("Name")
            .with_hint(PropertyHint::Required)
            .with_hint(PropertyHint::String);

        assert!(prop.is_required());
        assert!(prop.hints.contains(&PropertyHint::String));
    }

    #[test]
    fn test_design_steps() {
        let mut design = ShapeDesign::new("test");
        assert_eq!(design.current_step.number(), 1);

        design.next_step();
        assert_eq!(design.current_step.number(), 2);

        design.prev_step();
        assert_eq!(design.current_step.number(), 1);
    }

    #[test]
    fn test_design_validation() {
        let mut design = ShapeDesign::new("test");
        let issues = design.validate();

        // Should have warnings for no targets and no properties
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_shape_designer() {
        let mut designer = ShapeDesigner::new();

        let design = designer.new_design("ex:TestShape");
        design.add_target_class("ex:Person");
        design.add_property(PropertyDesign::new("ex:name").with_hint(PropertyHint::Required));

        assert_eq!(designer.list_designs().len(), 1);

        let shape = designer
            .get_design("ex:TestShape")
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");
        assert!(!shape.targets.is_empty());
    }

    #[test]
    fn test_completion_percentage() {
        let mut design = ShapeDesign::new("test")
            .with_label("Test")
            .with_description("A test shape");

        design.add_target_class("ex:Test");
        design.add_property(PropertyDesign::new("ex:prop1"));

        let percentage = design.completion_percentage();
        assert!(percentage > 0);
    }

    #[test]
    fn test_domain_info() {
        for domain in Domain::all() {
            assert!(!domain.name().is_empty());
            assert!(!domain.description().is_empty());
        }
    }
}
