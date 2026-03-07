//! Model Scaffolding and Templates
//!
//! This module provides pre-built templates and scaffolding utilities for quickly
//! creating common SAMM model patterns. It helps developers bootstrap new models
//! without writing boilerplate code.
//!
//! # Features
//!
//! - **Pre-built Templates**: Common model patterns (IoT sensor, product catalog, etc.)
//! - **Template Customization**: Parameterize templates with custom values
//! - **Template Composition**: Combine multiple templates
//! - **Quick Scaffolding**: Generate complete models from simple descriptions
//!
//! # Examples
//!
//! ```rust
//! use oxirs_samm::templates::scaffolding::{ModelTemplate, TemplateRegistry};
//!
//! // Use a pre-built IoT sensor template
//! let mut registry = TemplateRegistry::default();
//! let template = registry.get("iot_sensor").expect("key should exist");
//!
//! let aspect = template.instantiate("TemperatureSensor", "org.example", "1.0.0")
//!     .with_property("temperature", "xsd:decimal", false)
//!     .with_property("timestamp", "xsd:dateTime", false)
//!     .build();
//! ```

use crate::metamodel::{
    Aspect, Characteristic, CharacteristicKind, Entity, ModelElement, Operation, Property,
};
use std::collections::HashMap;

/// Function type for template builder construction
pub type TemplateBuilderFn = Box<dyn Fn(&str, &str, &str) -> TemplateBuilder>;

/// Template builder for creating SAMM models
pub struct TemplateBuilder {
    aspect: Aspect,
    property_configs: Vec<PropertyConfig>,
}

/// Configuration for a property in a template
struct PropertyConfig {
    name: String,
    data_type: String,
    optional: bool,
    is_collection: bool,
    description: Option<String>,
}

impl TemplateBuilder {
    /// Create a new template builder
    pub fn new(name: &str, namespace: &str, version: &str) -> Self {
        let urn = format!("urn:samm:{}:{}#{}", namespace, version, name);
        let aspect = Aspect::new(urn);

        Self {
            aspect,
            property_configs: Vec::new(),
        }
    }

    /// Add a property to the template
    pub fn with_property(mut self, name: &str, data_type: &str, optional: bool) -> Self {
        self.property_configs.push(PropertyConfig {
            name: name.to_string(),
            data_type: data_type.to_string(),
            optional,
            is_collection: false,
            description: None,
        });
        self
    }

    /// Add a collection property to the template
    pub fn with_collection(mut self, name: &str, data_type: &str, optional: bool) -> Self {
        self.property_configs.push(PropertyConfig {
            name: name.to_string(),
            data_type: data_type.to_string(),
            optional,
            is_collection: true,
            description: None,
        });
        self
    }

    /// Add a property with description
    pub fn with_described_property(
        mut self,
        name: &str,
        data_type: &str,
        optional: bool,
        description: &str,
    ) -> Self {
        self.property_configs.push(PropertyConfig {
            name: name.to_string(),
            data_type: data_type.to_string(),
            optional,
            is_collection: false,
            description: Some(description.to_string()),
        });
        self
    }

    /// Set aspect metadata
    pub fn with_description(mut self, description: &str) -> Self {
        self.aspect
            .metadata
            .add_description("en".to_string(), description.to_string());
        self
    }

    /// Set aspect preferred name
    pub fn with_preferred_name(mut self, name: &str) -> Self {
        self.aspect
            .metadata
            .add_preferred_name("en".to_string(), name.to_string());
        self
    }

    /// Build the final aspect
    pub fn build(mut self) -> Aspect {
        let namespace = self
            .aspect
            .urn()
            .split('#')
            .next()
            .unwrap_or("")
            .to_string();

        for config in self.property_configs {
            let prop_urn = format!("{}#{}", namespace, config.name);
            let mut prop = Property::new(prop_urn);

            let char_urn = format!("{}#{}Characteristic", namespace, config.name);
            let mut char = Characteristic::new(char_urn, CharacteristicKind::Trait);
            char.data_type = Some(config.data_type);

            prop.characteristic = Some(char);
            prop.optional = config.optional;
            prop.is_collection = config.is_collection;

            if let Some(desc) = config.description {
                prop.metadata.add_description("en".to_string(), desc);
            }

            self.aspect.add_property(prop);
        }

        self.aspect
    }
}

/// A reusable model template
pub struct ModelTemplate {
    name: String,
    description: String,
    builder_fn: TemplateBuilderFn,
}

impl ModelTemplate {
    /// Create a new template
    pub fn new<F>(name: &str, description: &str, builder_fn: F) -> Self
    where
        F: Fn(&str, &str, &str) -> TemplateBuilder + 'static,
    {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            builder_fn: Box::new(builder_fn),
        }
    }

    /// Instantiate the template with custom values
    pub fn instantiate(&self, name: &str, namespace: &str, version: &str) -> TemplateBuilder {
        (self.builder_fn)(name, namespace, version)
    }

    /// Get template name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get template description
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Registry of available templates
pub struct TemplateRegistry {
    templates: HashMap<String, ModelTemplate>,
}

impl TemplateRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    /// Register a template
    pub fn register(&mut self, template: ModelTemplate) {
        self.templates.insert(template.name().to_string(), template);
    }

    /// Get a template by name
    pub fn get(&self, name: &str) -> Option<&ModelTemplate> {
        self.templates.get(name)
    }

    /// List all available templates
    pub fn list(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }

    /// Get number of registered templates
    pub fn count(&self) -> usize {
        self.templates.len()
    }
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        let mut registry = Self::new();

        // IoT Sensor Template
        registry.register(ModelTemplate::new(
            "iot_sensor",
            "Template for IoT sensor devices with measurements and status",
            |name, namespace, version| {
                TemplateBuilder::new(name, namespace, version)
                    .with_preferred_name(name)
                    .with_description("IoT sensor aspect model")
                    .with_described_property(
                        "sensorId",
                        "xsd:string",
                        false,
                        "Unique sensor identifier",
                    )
                    .with_described_property("value", "xsd:decimal", false, "Measured value")
                    .with_described_property("unit", "xsd:string", false, "Unit of measurement")
                    .with_described_property(
                        "timestamp",
                        "xsd:dateTime",
                        false,
                        "Measurement timestamp",
                    )
                    .with_described_property("status", "xsd:string", true, "Sensor status")
            },
        ));

        // Product Catalog Template
        registry.register(ModelTemplate::new(
            "product_catalog",
            "Template for product catalog with basic product information",
            |name, namespace, version| {
                TemplateBuilder::new(name, namespace, version)
                    .with_preferred_name(name)
                    .with_description("Product catalog aspect model")
                    .with_described_property("productId", "xsd:string", false, "Product identifier")
                    .with_described_property("name", "xsd:string", false, "Product name")
                    .with_described_property(
                        "description",
                        "xsd:string",
                        true,
                        "Product description",
                    )
                    .with_described_property("price", "xsd:decimal", false, "Product price")
                    .with_described_property("currency", "xsd:string", false, "Price currency")
                    .with_described_property("inStock", "xsd:boolean", false, "Availability status")
                    .with_collection("categories", "xsd:string", true)
            },
        ));

        // Vehicle Tracking Template
        registry.register(ModelTemplate::new(
            "vehicle_tracking",
            "Template for vehicle tracking and telematics",
            |name, namespace, version| {
                TemplateBuilder::new(name, namespace, version)
                    .with_preferred_name(name)
                    .with_description("Vehicle tracking aspect model")
                    .with_described_property("vehicleId", "xsd:string", false, "Vehicle identifier")
                    .with_described_property("latitude", "xsd:decimal", false, "GPS latitude")
                    .with_described_property("longitude", "xsd:decimal", false, "GPS longitude")
                    .with_described_property("speed", "xsd:decimal", true, "Current speed")
                    .with_described_property("heading", "xsd:decimal", true, "Direction in degrees")
                    .with_described_property(
                        "timestamp",
                        "xsd:dateTime",
                        false,
                        "Position timestamp",
                    )
                    .with_described_property(
                        "fuelLevel",
                        "xsd:decimal",
                        true,
                        "Fuel level percentage",
                    )
            },
        ));

        // User Profile Template
        registry.register(ModelTemplate::new(
            "user_profile",
            "Template for user profile and account information",
            |name, namespace, version| {
                TemplateBuilder::new(name, namespace, version)
                    .with_preferred_name(name)
                    .with_description("User profile aspect model")
                    .with_described_property("userId", "xsd:string", false, "User identifier")
                    .with_described_property("username", "xsd:string", false, "Username")
                    .with_described_property("email", "xsd:string", false, "Email address")
                    .with_described_property("firstName", "xsd:string", true, "First name")
                    .with_described_property("lastName", "xsd:string", true, "Last name")
                    .with_described_property(
                        "createdAt",
                        "xsd:dateTime",
                        false,
                        "Account creation date",
                    )
                    .with_described_property(
                        "isActive",
                        "xsd:boolean",
                        false,
                        "Account active status",
                    )
            },
        ));

        // Time Series Data Template
        registry.register(ModelTemplate::new(
            "time_series",
            "Template for time series data with measurements",
            |name, namespace, version| {
                TemplateBuilder::new(name, namespace, version)
                    .with_preferred_name(name)
                    .with_description("Time series data aspect model")
                    .with_described_property(
                        "timestamp",
                        "xsd:dateTime",
                        false,
                        "Data point timestamp",
                    )
                    .with_described_property("value", "xsd:decimal", false, "Measured value")
                    .with_described_property("metricName", "xsd:string", false, "Metric name")
                    .with_described_property("tags", "xsd:string", true, "Metadata tags")
                    .with_described_property("quality", "xsd:decimal", true, "Data quality score")
            },
        ));

        registry
    }
}

/// Quick scaffolding utilities
pub mod quick {
    use super::*;

    /// Create a simple aspect with basic properties
    pub fn simple_aspect(name: &str, namespace: &str, version: &str) -> TemplateBuilder {
        TemplateBuilder::new(name, namespace, version)
    }

    /// Create an IoT sensor aspect
    pub fn iot_sensor(name: &str, namespace: &str, version: &str) -> Aspect {
        TemplateBuilder::new(name, namespace, version)
            .with_property("value", "xsd:decimal", false)
            .with_property("timestamp", "xsd:dateTime", false)
            .with_property("unit", "xsd:string", false)
            .build()
    }

    /// Create a data entity aspect
    pub fn data_entity(name: &str, namespace: &str, version: &str, id_field: &str) -> Aspect {
        TemplateBuilder::new(name, namespace, version)
            .with_property(id_field, "xsd:string", false)
            .with_property("createdAt", "xsd:dateTime", false)
            .with_property("updatedAt", "xsd:dateTime", true)
            .build()
    }

    /// Create a measurement aspect
    pub fn measurement(name: &str, namespace: &str, version: &str, value_type: &str) -> Aspect {
        TemplateBuilder::new(name, namespace, version)
            .with_property("value", value_type, false)
            .with_property("timestamp", "xsd:dateTime", false)
            .with_property("unit", "xsd:string", true)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::ModelElement;

    #[test]
    fn test_template_builder_basic() {
        let aspect = TemplateBuilder::new("TestAspect", "org.example", "1.0.0")
            .with_property("prop1", "xsd:string", false)
            .with_property("prop2", "xsd:integer", true)
            .build();

        assert_eq!(aspect.properties().len(), 2);
        assert!(!aspect.properties()[0].optional);
        assert!(aspect.properties()[1].optional);
    }

    #[test]
    fn test_template_builder_with_collection() {
        let aspect = TemplateBuilder::new("TestAspect", "org.example", "1.0.0")
            .with_collection("tags", "xsd:string", true)
            .build();

        assert_eq!(aspect.properties().len(), 1);
        assert!(aspect.properties()[0].is_collection);
    }

    #[test]
    fn test_template_builder_with_metadata() {
        let aspect = TemplateBuilder::new("TestAspect", "org.example", "1.0.0")
            .with_preferred_name("Test Aspect")
            .with_description("A test aspect")
            .build();

        assert_eq!(
            aspect.metadata.get_preferred_name("en"),
            Some("Test Aspect")
        );
        assert_eq!(aspect.metadata.get_description("en"), Some("A test aspect"));
    }

    #[test]
    fn test_template_registry_default() {
        let registry = TemplateRegistry::default();

        assert!(registry.count() >= 5);
        assert!(registry.get("iot_sensor").is_some());
        assert!(registry.get("product_catalog").is_some());
        assert!(registry.get("vehicle_tracking").is_some());
        assert!(registry.get("user_profile").is_some());
        assert!(registry.get("time_series").is_some());
    }

    #[test]
    fn test_template_instantiation() {
        let registry = TemplateRegistry::default();
        let template = registry.get("iot_sensor").expect("key should exist");

        let aspect = template
            .instantiate("TemperatureSensor", "org.example", "1.0.0")
            .build();

        assert!(aspect.properties().len() >= 4);
        assert!(aspect.properties().iter().any(|p| p.name() == "sensorId"));
        assert!(aspect.properties().iter().any(|p| p.name() == "value"));
    }

    #[test]
    fn test_template_list() {
        let registry = TemplateRegistry::default();
        let templates = registry.list();

        assert!(!templates.is_empty());
        assert!(templates.contains(&"iot_sensor"));
    }

    #[test]
    fn test_quick_iot_sensor() {
        let aspect = quick::iot_sensor("MySensor", "org.example", "1.0.0");

        assert_eq!(aspect.properties().len(), 3);
        assert!(aspect.properties().iter().any(|p| p.name() == "value"));
        assert!(aspect.properties().iter().any(|p| p.name() == "timestamp"));
        assert!(aspect.properties().iter().any(|p| p.name() == "unit"));
    }

    #[test]
    fn test_quick_data_entity() {
        let aspect = quick::data_entity("User", "org.example", "1.0.0", "userId");

        assert_eq!(aspect.properties().len(), 3);
        assert!(aspect.properties().iter().any(|p| p.name() == "userId"));
        assert!(aspect.properties().iter().any(|p| p.name() == "createdAt"));
    }

    #[test]
    fn test_quick_measurement() {
        let aspect = quick::measurement("Temperature", "org.example", "1.0.0", "xsd:decimal");

        assert_eq!(aspect.properties().len(), 3);
        assert!(aspect.properties().iter().any(|p| p.name() == "value"));
    }

    #[test]
    fn test_custom_template_registration() {
        let mut registry = TemplateRegistry::new();

        let custom = ModelTemplate::new("custom", "Custom template", |name, namespace, version| {
            TemplateBuilder::new(name, namespace, version).with_property(
                "customProp",
                "xsd:string",
                false,
            )
        });

        registry.register(custom);

        assert_eq!(registry.count(), 1);
        assert!(registry.get("custom").is_some());
    }

    #[test]
    fn test_described_property() {
        let aspect = TemplateBuilder::new("Test", "org.example", "1.0.0")
            .with_described_property("prop1", "xsd:string", false, "A test property")
            .build();

        assert_eq!(aspect.properties().len(), 1);
        assert_eq!(
            aspect.properties()[0].metadata.get_description("en"),
            Some("A test property")
        );
    }
}
