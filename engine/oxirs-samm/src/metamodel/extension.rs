//! SAMM Extension Support
//!
//! This module provides support for extending the SAMM metamodel with custom elements.
//! Extensions allow users to add domain-specific concepts while maintaining compatibility
//! with the core SAMM specification.
//!
//! # Use Cases
//!
//! - **Domain-specific vocabularies**: Add industry-specific terminology
//! - **Custom characteristics**: Define specialized data validation rules
//! - **Extended properties**: Add metadata beyond SAMM's standard set
//! - **Organization standards**: Enforce company-specific modeling guidelines
//!
//! # Example
//!
//! ```rust
//! use oxirs_samm::metamodel::extension::{Extension, ExtensionElement, ExtensionRegistry};
//!
//! // Create a custom extension
//! let mut extension = Extension::new(
//!     "urn:extension:org.example:automotive:1.0.0",
//!     "Automotive Extension",
//! );
//!
//! extension.set_description("Automotive industry-specific SAMM extensions");
//! extension.add_author("Automotive Standards Organization");
//!
//! // Register custom elements
//! extension.add_element(ExtensionElement::new(
//!     "VehicleCharacteristic",
//!     "Custom characteristic for vehicle data",
//! ));
//!
//! // Use the extension registry
//! let mut registry = ExtensionRegistry::new();
//! registry.register(extension);
//!
//! assert_eq!(registry.count(), 1);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// An extension to the SAMM metamodel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Extension {
    /// Unique namespace URI for this extension
    pub namespace: String,
    /// Human-readable name
    pub name: String,
    /// Description of the extension
    pub description: Option<String>,
    /// Version of the extension
    pub version: Option<String>,
    /// Authors/maintainers
    pub authors: Vec<String>,
    /// Base SAMM version this extends
    pub samm_version: String,
    /// Custom elements defined by this extension
    pub elements: Vec<ExtensionElement>,
    /// Custom properties that can be added to standard SAMM elements
    pub custom_properties: HashMap<String, PropertyDefinition>,
    /// Custom validation rules
    pub validation_rules: Vec<ValidationRule>,
}

impl Extension {
    /// Create a new extension
    ///
    /// # Arguments
    ///
    /// * `namespace` - Unique namespace URI (e.g., "urn:extension:org.example:domain:1.0.0")
    /// * `name` - Human-readable name
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_samm::metamodel::extension::Extension;
    ///
    /// let ext = Extension::new(
    ///     "urn:extension:org.example:iot:1.0.0",
    ///     "IoT Extension",
    /// );
    /// assert_eq!(ext.name, "IoT Extension");
    /// ```
    pub fn new(namespace: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
            description: None,
            version: Some("1.0.0".to_string()),
            authors: Vec::new(),
            samm_version: "2.1.0".to_string(),
            elements: Vec::new(),
            custom_properties: HashMap::new(),
            validation_rules: Vec::new(),
        }
    }

    /// Set the extension description
    pub fn set_description(&mut self, description: impl Into<String>) {
        self.description = Some(description.into());
    }

    /// Set the extension version
    pub fn set_version(&mut self, version: impl Into<String>) {
        self.version = Some(version.into());
    }

    /// Add an author
    pub fn add_author(&mut self, author: impl Into<String>) {
        self.authors.push(author.into());
    }

    /// Set the base SAMM version
    pub fn set_samm_version(&mut self, version: impl Into<String>) {
        self.samm_version = version.into();
    }

    /// Add a custom element to this extension
    pub fn add_element(&mut self, element: ExtensionElement) {
        self.elements.push(element);
    }

    /// Add a custom property definition
    pub fn add_custom_property(&mut self, name: String, property: PropertyDefinition) {
        self.custom_properties.insert(name, property);
    }

    /// Add a validation rule
    pub fn add_validation_rule(&mut self, rule: ValidationRule) {
        self.validation_rules.push(rule);
    }

    /// Get a custom element by name
    pub fn get_element(&self, name: &str) -> Option<&ExtensionElement> {
        self.elements.iter().find(|e| e.name == name)
    }

    /// Check if this extension defines a custom element
    pub fn has_element(&self, name: &str) -> bool {
        self.elements.iter().any(|e| e.name == name)
    }
}

/// A custom element defined in an extension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionElement {
    /// Element name
    pub name: String,
    /// Element description
    pub description: String,
    /// Element type (e.g., "Characteristic", "Constraint", "Entity")
    pub element_type: String,
    /// Parent type this element extends
    pub extends: Option<String>,
    /// Required properties
    pub required_properties: Vec<String>,
    /// Optional properties
    pub optional_properties: Vec<String>,
    /// Custom attributes
    pub attributes: HashMap<String, String>,
}

impl ExtensionElement {
    /// Create a new extension element
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            element_type: "Custom".to_string(),
            extends: None,
            required_properties: Vec::new(),
            optional_properties: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    /// Set the element type
    pub fn with_type(mut self, element_type: impl Into<String>) -> Self {
        self.element_type = element_type.into();
        self
    }

    /// Set what this element extends
    pub fn extends(mut self, parent: impl Into<String>) -> Self {
        self.extends = Some(parent.into());
        self
    }

    /// Add a required property
    pub fn require(mut self, property: impl Into<String>) -> Self {
        self.required_properties.push(property.into());
        self
    }

    /// Add an optional property
    pub fn optional(mut self, property: impl Into<String>) -> Self {
        self.optional_properties.push(property.into());
        self
    }

    /// Add a custom attribute
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

/// Definition of a custom property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyDefinition {
    /// Property name
    pub name: String,
    /// Property description
    pub description: String,
    /// Data type
    pub data_type: String,
    /// Whether this property is required
    pub required: bool,
    /// Default value
    pub default_value: Option<String>,
    /// Allowed values (for enumerations)
    pub allowed_values: Vec<String>,
}

impl PropertyDefinition {
    /// Create a new property definition
    pub fn new(name: impl Into<String>, data_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            data_type: data_type.into(),
            required: false,
            default_value: None,
            allowed_values: Vec::new(),
        }
    }

    /// Set the description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Mark as required
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Set default value
    pub fn with_default(mut self, value: impl Into<String>) -> Self {
        self.default_value = Some(value.into());
        self
    }

    /// Add allowed value
    pub fn allow(mut self, value: impl Into<String>) -> Self {
        self.allowed_values.push(value.into());
        self
    }
}

/// A validation rule for extension elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Element type this rule applies to
    pub applies_to: String,
    /// Rule severity (Error, Warning, Info)
    pub severity: ValidationSeverity,
    /// Rule implementation (as a SHACL-like expression or custom code)
    pub expression: String,
}

/// Validation rule severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Error - model is invalid
    Error,
    /// Warning - model may have issues
    Warning,
    /// Info - informational message
    Info,
}

/// Registry for managing SAMM extensions
///
/// The registry is thread-safe and can be shared across threads.
pub struct ExtensionRegistry {
    extensions: Arc<RwLock<HashMap<String, Extension>>>,
}

impl ExtensionRegistry {
    /// Create a new extension registry
    pub fn new() -> Self {
        Self {
            extensions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register an extension
    ///
    /// # Arguments
    ///
    /// * `extension` - The extension to register
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_samm::metamodel::extension::{Extension, ExtensionRegistry};
    ///
    /// let mut registry = ExtensionRegistry::new();
    /// let ext = Extension::new("urn:ext:example", "Example");
    /// registry.register(ext);
    /// assert_eq!(registry.count(), 1);
    /// ```
    pub fn register(&self, extension: Extension) {
        let namespace = extension.namespace.clone();
        let mut extensions = self.extensions.write().unwrap();
        extensions.insert(namespace, extension);
    }

    /// Get an extension by namespace
    pub fn get(&self, namespace: &str) -> Option<Extension> {
        let extensions = self.extensions.read().unwrap();
        extensions.get(namespace).cloned()
    }

    /// Check if an extension is registered
    pub fn has(&self, namespace: &str) -> bool {
        let extensions = self.extensions.read().unwrap();
        extensions.contains_key(namespace)
    }

    /// Remove an extension
    pub fn remove(&self, namespace: &str) -> bool {
        let mut extensions = self.extensions.write().unwrap();
        extensions.remove(namespace).is_some()
    }

    /// List all registered extension namespaces
    pub fn list(&self) -> Vec<String> {
        let extensions = self.extensions.read().unwrap();
        extensions.keys().cloned().collect()
    }

    /// Get the number of registered extensions
    pub fn count(&self) -> usize {
        let extensions = self.extensions.read().unwrap();
        extensions.len()
    }

    /// Clear all extensions
    pub fn clear(&self) {
        let mut extensions = self.extensions.write().unwrap();
        extensions.clear();
    }

    /// Get all extensions
    pub fn all(&self) -> Vec<Extension> {
        let extensions = self.extensions.read().unwrap();
        extensions.values().cloned().collect()
    }

    /// Find extensions that extend a specific SAMM version
    pub fn find_by_samm_version(&self, version: &str) -> Vec<Extension> {
        let extensions = self.extensions.read().unwrap();
        extensions
            .values()
            .filter(|ext| ext.samm_version == version)
            .cloned()
            .collect()
    }
}

impl Default for ExtensionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ExtensionRegistry {
    fn clone(&self) -> Self {
        Self {
            extensions: Arc::clone(&self.extensions),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_creation() {
        let ext = Extension::new("urn:ext:test", "Test Extension");
        assert_eq!(ext.namespace, "urn:ext:test");
        assert_eq!(ext.name, "Test Extension");
        assert_eq!(ext.samm_version, "2.1.0");
    }

    #[test]
    fn test_extension_builder() {
        let mut ext = Extension::new("urn:ext:test", "Test");
        ext.set_description("A test extension");
        ext.set_version("2.0.0");
        ext.add_author("Test Author");
        ext.set_samm_version("2.1.0");

        assert_eq!(ext.description, Some("A test extension".to_string()));
        assert_eq!(ext.version, Some("2.0.0".to_string()));
        assert_eq!(ext.authors.len(), 1);
    }

    #[test]
    fn test_extension_element() {
        let element = ExtensionElement::new("CustomChar", "Custom characteristic")
            .with_type("Characteristic")
            .extends("samm:Characteristic")
            .require("customProperty")
            .optional("optionalProperty")
            .attribute("category", "validation");

        assert_eq!(element.name, "CustomChar");
        assert_eq!(element.element_type, "Characteristic");
        assert_eq!(element.extends, Some("samm:Characteristic".to_string()));
        assert_eq!(element.required_properties.len(), 1);
        assert_eq!(element.optional_properties.len(), 1);
        assert_eq!(
            element.attributes.get("category"),
            Some(&"validation".to_string())
        );
    }

    #[test]
    fn test_property_definition() {
        let prop = PropertyDefinition::new("industryCode", "string")
            .with_description("Industry classification code")
            .required()
            .with_default("UNKNOWN")
            .allow("AUTOMOTIVE")
            .allow("AEROSPACE")
            .allow("MANUFACTURING");

        assert_eq!(prop.name, "industryCode");
        assert_eq!(prop.data_type, "string");
        assert!(prop.required);
        assert_eq!(prop.default_value, Some("UNKNOWN".to_string()));
        assert_eq!(prop.allowed_values.len(), 3);
    }

    #[test]
    fn test_registry_basic() {
        let registry = ExtensionRegistry::new();
        assert_eq!(registry.count(), 0);

        let ext = Extension::new("urn:ext:test1", "Test 1");
        registry.register(ext);
        assert_eq!(registry.count(), 1);

        assert!(registry.has("urn:ext:test1"));
        assert!(!registry.has("urn:ext:nonexistent"));
    }

    #[test]
    fn test_registry_get() {
        let registry = ExtensionRegistry::new();
        let ext = Extension::new("urn:ext:test", "Test");
        registry.register(ext);

        let retrieved = registry.get("urn:ext:test");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Test");
    }

    #[test]
    fn test_registry_remove() {
        let registry = ExtensionRegistry::new();
        let ext = Extension::new("urn:ext:test", "Test");
        registry.register(ext);

        assert_eq!(registry.count(), 1);
        assert!(registry.remove("urn:ext:test"));
        assert_eq!(registry.count(), 0);
        assert!(!registry.remove("urn:ext:test"));
    }

    #[test]
    fn test_registry_list() {
        let registry = ExtensionRegistry::new();
        registry.register(Extension::new("urn:ext:a", "A"));
        registry.register(Extension::new("urn:ext:b", "B"));
        registry.register(Extension::new("urn:ext:c", "C"));

        let list = registry.list();
        assert_eq!(list.len(), 3);
        assert!(list.contains(&"urn:ext:a".to_string()));
        assert!(list.contains(&"urn:ext:b".to_string()));
        assert!(list.contains(&"urn:ext:c".to_string()));
    }

    #[test]
    fn test_registry_clear() {
        let registry = ExtensionRegistry::new();
        registry.register(Extension::new("urn:ext:a", "A"));
        registry.register(Extension::new("urn:ext:b", "B"));

        assert_eq!(registry.count(), 2);
        registry.clear();
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_find_by_samm_version() {
        let registry = ExtensionRegistry::new();

        let mut ext1 = Extension::new("urn:ext:a", "A");
        ext1.set_samm_version("2.1.0");
        registry.register(ext1);

        let mut ext2 = Extension::new("urn:ext:b", "B");
        ext2.set_samm_version("2.2.0");
        registry.register(ext2);

        let mut ext3 = Extension::new("urn:ext:c", "C");
        ext3.set_samm_version("2.1.0");
        registry.register(ext3);

        let v21_exts = registry.find_by_samm_version("2.1.0");
        assert_eq!(v21_exts.len(), 2);

        let v22_exts = registry.find_by_samm_version("2.2.0");
        assert_eq!(v22_exts.len(), 1);
    }

    #[test]
    fn test_extension_add_element() {
        let mut ext = Extension::new("urn:ext:test", "Test");
        let element = ExtensionElement::new("Custom", "Custom element");
        ext.add_element(element);

        assert_eq!(ext.elements.len(), 1);
        assert!(ext.has_element("Custom"));
        assert!(!ext.has_element("NonExistent"));
    }

    #[test]
    fn test_extension_custom_property() {
        let mut ext = Extension::new("urn:ext:test", "Test");
        let prop = PropertyDefinition::new("custom", "string");
        ext.add_custom_property("customProp".to_string(), prop);

        assert_eq!(ext.custom_properties.len(), 1);
        assert!(ext.custom_properties.contains_key("customProp"));
    }
}
