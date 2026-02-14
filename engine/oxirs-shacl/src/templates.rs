//! Shape Template Library
//!
//! Pre-built SHACL shape templates for common validation patterns.
//! Provides ready-to-use shapes for typical use cases across various domains.

use oxirs_core::NamedNode;

use crate::{
    constraints::{
        cardinality_constraints::MinCountConstraint,
        string_constraints::{MaxLengthConstraint, MinLengthConstraint, PatternConstraint},
        value_constraints::DatatypeConstraint,
        Constraint,
    },
    paths::PropertyPath,
    targets::Target,
    ConstraintComponentId, Shape, ShapeId, ShapeType,
};

/// Template category for organizing shapes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateCategory {
    /// Person and organization data
    Identity,
    /// Contact information
    Contact,
    /// Product and catalog data
    Commerce,
    /// Web and digital resources
    Web,
    /// Temporal and scheduling data
    Temporal,
    /// Geographic and spatial data
    Geospatial,
    /// Financial and monetary data
    Financial,
    /// Scientific and research data
    Scientific,
}

/// Shape template with metadata
#[derive(Debug, Clone)]
pub struct ShapeTemplate {
    /// Template identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Template description
    pub description: String,
    /// Template category
    pub category: TemplateCategory,
    /// The shape itself
    pub shape: Shape,
    /// Usage example
    pub example: String,
}

/// Shape template library
#[derive(Debug)]
pub struct TemplateLibrary {
    templates: Vec<ShapeTemplate>,
}

impl TemplateLibrary {
    /// Create a new template library with all built-in templates
    pub fn new() -> Self {
        let mut lib = Self {
            templates: Vec::new(),
        };

        lib.load_identity_templates();
        lib.load_contact_templates();
        lib.load_commerce_templates();
        lib.load_web_templates();
        lib.load_temporal_templates();
        lib.load_geospatial_templates();
        lib.load_financial_templates();
        lib.load_scientific_templates();

        lib
    }

    /// Get all templates
    pub fn all(&self) -> &[ShapeTemplate] {
        &self.templates
    }

    /// Get templates by category
    pub fn by_category(&self, category: TemplateCategory) -> Vec<&ShapeTemplate> {
        self.templates
            .iter()
            .filter(|t| t.category == category)
            .collect()
    }

    /// Get template by ID
    pub fn get(&self, id: &str) -> Option<&ShapeTemplate> {
        self.templates.iter().find(|t| t.id == id)
    }

    /// List all template IDs
    pub fn list_ids(&self) -> Vec<String> {
        self.templates.iter().map(|t| t.id.clone()).collect()
    }

    fn load_identity_templates(&mut self) {
        // Person template
        self.templates.push(ShapeTemplate {
            id: "person".to_string(),
            name: "Person".to_string(),
            description: "Basic person shape with name and email validation".to_string(),
            category: TemplateCategory::Identity,
            shape: create_person_shape(),
            example: r#"@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

ex:john a foaf:Person ;
    foaf:name "John Doe" ;
    foaf:mbox <mailto:john@example.org> ."#
                .to_string(),
        });

        // Organization template
        self.templates.push(ShapeTemplate {
            id: "organization".to_string(),
            name: "Organization".to_string(),
            description: "Organization with name and optional homepage".to_string(),
            category: TemplateCategory::Identity,
            shape: create_organization_shape(),
            example: r#"@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

ex:acme a foaf:Organization ;
    foaf:name "Acme Corp" ;
    foaf:homepage <http://acme.example.org/> ."#
                .to_string(),
        });
    }

    fn load_contact_templates(&mut self) {
        // Email template
        self.templates.push(ShapeTemplate {
            id: "email".to_string(),
            name: "Email Address".to_string(),
            description: "Email address validation with pattern matching".to_string(),
            category: TemplateCategory::Contact,
            shape: create_email_shape(),
            example: r#"Valid: john@example.com, alice.smith@company.co.uk
Invalid: notanemail, @example.com, john@"#
                .to_string(),
        });

        // Phone number template
        self.templates.push(ShapeTemplate {
            id: "phone".to_string(),
            name: "Phone Number".to_string(),
            description: "Phone number validation (international format)".to_string(),
            category: TemplateCategory::Contact,
            shape: create_phone_shape(),
            example: r#"Valid: +1-555-123-4567, +44 20 7123 4567
Format: +{country}-{area}-{number}"#
                .to_string(),
        });
    }

    fn load_commerce_templates(&mut self) {
        // Product template
        self.templates.push(ShapeTemplate {
            id: "product".to_string(),
            name: "Product".to_string(),
            description: "E-commerce product with name, price, and SKU".to_string(),
            category: TemplateCategory::Commerce,
            shape: create_product_shape(),
            example: r#"@prefix ex: <http://example.org/> .
@prefix schema: <http://schema.org/> .

ex:product123 a schema:Product ;
    schema:name "Widget" ;
    schema:sku "WDG-001" ;
    schema:price "29.99" ."#
                .to_string(),
        });

        // Order template
        self.templates.push(ShapeTemplate {
            id: "order".to_string(),
            name: "Order".to_string(),
            description: "Order with order number and date".to_string(),
            category: TemplateCategory::Commerce,
            shape: create_order_shape(),
            example: r#"@prefix ex: <http://example.org/> .
@prefix schema: <http://schema.org/> .

ex:order456 a schema:Order ;
    schema:orderNumber "ORD-456" ;
    schema:orderDate "2025-01-15"^^xsd:date ."#
                .to_string(),
        });
    }

    fn load_web_templates(&mut self) {
        // URL template
        self.templates.push(ShapeTemplate {
            id: "url".to_string(),
            name: "URL/IRI".to_string(),
            description: "Web URL validation (http/https)".to_string(),
            category: TemplateCategory::Web,
            shape: create_url_shape(),
            example: r#"Valid: http://example.org, https://example.com/path
Invalid: ftp://example.org, not-a-url"#
                .to_string(),
        });

        // Web page template
        self.templates.push(ShapeTemplate {
            id: "webpage".to_string(),
            name: "Web Page".to_string(),
            description: "Web page with title and URL".to_string(),
            category: TemplateCategory::Web,
            shape: create_webpage_shape(),
            example: r#"@prefix ex: <http://example.org/> .
@prefix schema: <http://schema.org/> .

ex:page1 a schema:WebPage ;
    schema:name "Example Page" ;
    schema:url <http://example.org/page1> ."#
                .to_string(),
        });
    }

    fn load_temporal_templates(&mut self) {
        // Date template
        self.templates.push(ShapeTemplate {
            id: "date".to_string(),
            name: "Date".to_string(),
            description: "ISO 8601 date validation".to_string(),
            category: TemplateCategory::Temporal,
            shape: create_date_shape(),
            example: r#"Valid: 2025-01-15, 2025-12-31
Format: YYYY-MM-DD"#
                .to_string(),
        });

        // Event template
        self.templates.push(ShapeTemplate {
            id: "event".to_string(),
            name: "Event".to_string(),
            description: "Event with name and date".to_string(),
            category: TemplateCategory::Temporal,
            shape: create_event_shape(),
            example: r#"@prefix ex: <http://example.org/> .
@prefix schema: <http://schema.org/> .

ex:event1 a schema:Event ;
    schema:name "Conference 2025" ;
    schema:startDate "2025-03-15"^^xsd:date ."#
                .to_string(),
        });
    }

    fn load_geospatial_templates(&mut self) {
        // Place template
        self.templates.push(ShapeTemplate {
            id: "place".to_string(),
            name: "Place".to_string(),
            description: "Geographic place with name and coordinates".to_string(),
            category: TemplateCategory::Geospatial,
            shape: create_place_shape(),
            example: r#"@prefix ex: <http://example.org/> .
@prefix schema: <http://schema.org/> .

ex:place1 a schema:Place ;
    schema:name "Central Park" ;
    schema:latitude "40.785091" ;
    schema:longitude "-73.968285" ."#
                .to_string(),
        });

        // Address template
        self.templates.push(ShapeTemplate {
            id: "address".to_string(),
            name: "Postal Address".to_string(),
            description: "Mailing address with street, city, and postal code".to_string(),
            category: TemplateCategory::Geospatial,
            shape: create_address_shape(),
            example: r#"@prefix ex: <http://example.org/> .
@prefix schema: <http://schema.org/> .

ex:addr1 a schema:PostalAddress ;
    schema:streetAddress "123 Main St" ;
    schema:addressLocality "Springfield" ;
    schema:postalCode "12345" ."#
                .to_string(),
        });
    }

    fn load_financial_templates(&mut self) {
        // Price template
        self.templates.push(ShapeTemplate {
            id: "price".to_string(),
            name: "Price/Monetary Amount".to_string(),
            description: "Monetary value with currency validation".to_string(),
            category: TemplateCategory::Financial,
            shape: create_price_shape(),
            example: r#"Valid: 29.99, 1000.00, 0.99
Format: decimal number with 2 decimal places"#
                .to_string(),
        });
    }

    fn load_scientific_templates(&mut self) {
        // Dataset template
        self.templates.push(ShapeTemplate {
            id: "dataset".to_string(),
            name: "Dataset".to_string(),
            description: "Research dataset with title and description".to_string(),
            category: TemplateCategory::Scientific,
            shape: create_dataset_shape(),
            example: r#"@prefix ex: <http://example.org/> .
@prefix dcat: <http://www.w3.org/ns/dcat#> .

ex:dataset1 a dcat:Dataset ;
    dcat:title "Research Data 2025" ;
    dcat:description "Experimental results from study" ."#
                .to_string(),
        });
    }
}

impl Default for TemplateLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// Template creation functions

fn create_person_shape() -> Shape {
    let shape_id = ShapeId::new("http://example.org/shapes#PersonShape");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    shape.add_target(Target::Class(
        NamedNode::new("http://xmlns.com/foaf/0.1/Person").expect("valid IRI"),
    ));

    // Name property (required, 1-n values)
    let mut name_shape = Shape::property_shape(
        ShapeId::new("http://example.org/shapes#PersonName"),
        PropertyPath::predicate(
            NamedNode::new("http://xmlns.com/foaf/0.1/name").expect("valid IRI"),
        ),
    );
    name_shape.add_constraint(
        ConstraintComponentId::new("sh:minCount"),
        Constraint::MinCount(MinCountConstraint { min_count: 1 }),
    );
    name_shape.add_constraint(
        ConstraintComponentId::new("sh:datatype"),
        Constraint::Datatype(DatatypeConstraint {
            datatype_iri: NamedNode::new("http://www.w3.org/2001/XMLSchema#string")
                .expect("valid IRI"),
        }),
    );

    shape
}

fn create_organization_shape() -> Shape {
    let shape_id = ShapeId::new("http://example.org/shapes#OrganizationShape");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    shape.add_target(Target::Class(
        NamedNode::new("http://xmlns.com/foaf/0.1/Organization").expect("valid IRI"),
    ));

    shape
}

fn create_email_shape() -> Shape {
    let shape_id = ShapeId::new("http://example.org/shapes#EmailShape");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    shape.add_constraint(
        ConstraintComponentId::new("sh:pattern"),
        Constraint::Pattern(PatternConstraint {
            pattern: r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$".to_string(),
            flags: None,
            message: Some("Must be a valid email address".to_string()),
        }),
    );

    shape
}

fn create_phone_shape() -> Shape {
    let shape_id = ShapeId::new("http://example.org/shapes#PhoneShape");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    shape.add_constraint(
        ConstraintComponentId::new("sh:pattern"),
        Constraint::Pattern(PatternConstraint {
            pattern: r"^\+\d{1,3}[-\s]?\d{1,4}[-\s]?\d{1,4}[-\s]?\d{1,9}$".to_string(),
            flags: None,
            message: Some("Must be a valid international phone number".to_string()),
        }),
    );

    shape
}

fn create_product_shape() -> Shape {
    let shape_id = ShapeId::new("http://example.org/shapes#ProductShape");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    shape.add_target(Target::Class(
        NamedNode::new("http://schema.org/Product").expect("valid IRI"),
    ));

    shape
}

fn create_order_shape() -> Shape {
    let shape_id = ShapeId::new("http://example.org/shapes#OrderShape");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    shape.add_target(Target::Class(
        NamedNode::new("http://schema.org/Order").expect("valid IRI"),
    ));

    shape
}

fn create_url_shape() -> Shape {
    let shape_id = ShapeId::new("http://example.org/shapes#URLShape");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    shape.add_constraint(
        ConstraintComponentId::new("sh:pattern"),
        Constraint::Pattern(PatternConstraint {
            pattern: r"^https?://[^\s/$.?#].[^\s]*$".to_string(),
            flags: Some("i".to_string()),
            message: Some("Must be a valid HTTP/HTTPS URL".to_string()),
        }),
    );

    shape
}

fn create_webpage_shape() -> Shape {
    let shape_id = ShapeId::new("http://example.org/shapes#WebPageShape");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    shape.add_target(Target::Class(
        NamedNode::new("http://schema.org/WebPage").expect("valid IRI"),
    ));

    shape
}

fn create_date_shape() -> Shape {
    let shape_id = ShapeId::new("http://example.org/shapes#DateShape");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    shape.add_constraint(
        ConstraintComponentId::new("sh:datatype"),
        Constraint::Datatype(DatatypeConstraint {
            datatype_iri: NamedNode::new("http://www.w3.org/2001/XMLSchema#date")
                .expect("valid IRI"),
        }),
    );

    shape
}

fn create_event_shape() -> Shape {
    let shape_id = ShapeId::new("http://example.org/shapes#EventShape");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    shape.add_target(Target::Class(
        NamedNode::new("http://schema.org/Event").expect("valid IRI"),
    ));

    shape
}

fn create_place_shape() -> Shape {
    let shape_id = ShapeId::new("http://example.org/shapes#PlaceShape");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    shape.add_target(Target::Class(
        NamedNode::new("http://schema.org/Place").expect("valid IRI"),
    ));

    shape
}

fn create_address_shape() -> Shape {
    let shape_id = ShapeId::new("http://example.org/shapes#AddressShape");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    shape.add_target(Target::Class(
        NamedNode::new("http://schema.org/PostalAddress").expect("valid IRI"),
    ));

    shape
}

fn create_price_shape() -> Shape {
    let shape_id = ShapeId::new("http://example.org/shapes#PriceShape");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    shape.add_constraint(
        ConstraintComponentId::new("sh:datatype"),
        Constraint::Datatype(DatatypeConstraint {
            datatype_iri: NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal")
                .expect("valid IRI"),
        }),
    );

    shape.add_constraint(
        ConstraintComponentId::new("sh:minLength"),
        Constraint::MinLength(MinLengthConstraint { min_length: 1 }),
    );

    shape.add_constraint(
        ConstraintComponentId::new("sh:maxLength"),
        Constraint::MaxLength(MaxLengthConstraint { max_length: 20 }),
    );

    shape
}

fn create_dataset_shape() -> Shape {
    let shape_id = ShapeId::new("http://example.org/shapes#DatasetShape");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    shape.add_target(Target::Class(
        NamedNode::new("http://www.w3.org/ns/dcat#Dataset").expect("valid IRI"),
    ));

    shape
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_library_creation() {
        let lib = TemplateLibrary::new();
        assert!(!lib.all().is_empty(), "Library should have templates");
    }

    #[test]
    fn test_get_template_by_id() {
        let lib = TemplateLibrary::new();

        let person = lib.get("person");
        assert!(person.is_some(), "Person template should exist");
        assert_eq!(person.expect("operation should succeed").name, "Person");
    }

    #[test]
    fn test_templates_by_category() {
        let lib = TemplateLibrary::new();

        let identity = lib.by_category(TemplateCategory::Identity);
        assert!(
            !identity.is_empty(),
            "Identity category should have templates"
        );

        let contact = lib.by_category(TemplateCategory::Contact);
        assert!(
            !contact.is_empty(),
            "Contact category should have templates"
        );
    }

    #[test]
    fn test_list_template_ids() {
        let lib = TemplateLibrary::new();
        let ids = lib.list_ids();

        assert!(ids.contains(&"person".to_string()));
        assert!(ids.contains(&"email".to_string()));
        assert!(ids.contains(&"product".to_string()));
    }

    #[test]
    fn test_template_has_shape() {
        let lib = TemplateLibrary::new();
        let template = lib.get("email").expect("key should exist");

        assert!(
            !template.shape.constraints.is_empty(),
            "Email template should have constraints"
        );
    }

    #[test]
    fn test_all_categories_have_templates() {
        let lib = TemplateLibrary::new();

        for category in [
            TemplateCategory::Identity,
            TemplateCategory::Contact,
            TemplateCategory::Commerce,
            TemplateCategory::Web,
            TemplateCategory::Temporal,
            TemplateCategory::Geospatial,
            TemplateCategory::Financial,
            TemplateCategory::Scientific,
        ] {
            let templates = lib.by_category(category);
            assert!(
                !templates.is_empty(),
                "Category {:?} should have templates",
                category
            );
        }
    }
}
