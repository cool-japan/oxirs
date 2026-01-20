//! Property Model Element
//!
//! A Property defines a named feature of an Aspect.

use super::{Characteristic, ElementMetadata, ModelElement};
use serde::{Deserialize, Serialize};

/// A Property in the SAMM meta model
///
/// Properties are named features of Aspects that have defined characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Property {
    /// Element metadata (URN, names, descriptions)
    pub metadata: ElementMetadata,

    /// The characteristic that defines the semantics of this property
    pub characteristic: Option<Characteristic>,

    /// Optional: Example values
    pub example_values: Vec<String>,

    /// Whether this property is optional (default: false)
    pub optional: bool,

    /// Whether this property is a collection (default: false)
    pub is_collection: bool,

    /// Payload name (for serialization, if different from property name)
    pub payload_name: Option<String>,

    /// Whether this property is abstract (cannot be instantiated directly)
    pub is_abstract: bool,

    /// If this property extends another property
    pub extends: Option<String>,
}

impl Property {
    /// Create a new Property
    pub fn new(urn: String) -> Self {
        Self {
            metadata: ElementMetadata::new(urn),
            characteristic: None,
            example_values: Vec::new(),
            optional: false,
            is_collection: false,
            payload_name: None,
            is_abstract: false,
            extends: None,
        }
    }

    /// Set the characteristic of this property
    pub fn with_characteristic(mut self, characteristic: Characteristic) -> Self {
        self.characteristic = Some(characteristic);
        self
    }

    /// Mark this property as optional
    pub fn as_optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// Mark this property as a collection
    pub fn as_collection(mut self) -> Self {
        self.is_collection = true;
        self
    }

    /// Set the payload name
    pub fn with_payload_name(mut self, name: String) -> Self {
        self.payload_name = Some(name);
        self
    }

    /// Mark this property as abstract
    pub fn as_abstract(mut self) -> Self {
        self.is_abstract = true;
        self
    }

    /// Set the property this extends
    pub fn extends(mut self, property_urn: String) -> Self {
        self.extends = Some(property_urn);
        self
    }

    /// Get the effective name for serialization
    pub fn effective_name(&self) -> String {
        self.payload_name.clone().unwrap_or_else(|| self.name())
    }
}

impl ModelElement for Property {
    fn urn(&self) -> &str {
        &self.metadata.urn
    }

    fn metadata(&self) -> &ElementMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_creation() {
        let property = Property::new("urn:samm:org.example:1.0.0#testProperty".to_string())
            .as_optional()
            .with_payload_name("test_property".to_string());

        assert_eq!(property.name(), "testProperty");
        assert_eq!(property.effective_name(), "test_property");
        assert!(property.optional);
        assert!(!property.is_abstract);
    }
}
