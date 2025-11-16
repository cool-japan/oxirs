//! SAMM Meta Model Types
//!
//! This module defines the core types of the Semantic Aspect Meta Model.

mod aspect;
mod characteristic;
mod entity;
pub mod extension;
mod operation;
mod property;

pub use aspect::Aspect;
pub use characteristic::{Characteristic, CharacteristicKind, Constraint};
pub use entity::{ComplexType, Entity};
pub use extension::{
    Extension, ExtensionElement, ExtensionRegistry, PropertyDefinition, ValidationRule,
    ValidationSeverity,
};
pub use operation::{Event, Operation};
pub use property::Property;

use oxrdf::{NamedNode, Term};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Common metadata for all SAMM elements
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementMetadata {
    /// URN identifier
    pub urn: String,

    /// Preferred names in different languages
    pub preferred_names: HashMap<String, String>,

    /// Descriptions in different languages
    pub descriptions: HashMap<String, String>,

    /// See references (URLs)
    pub see_refs: Vec<String>,
}

impl ElementMetadata {
    /// Create new element metadata
    pub fn new(urn: String) -> Self {
        Self {
            urn,
            preferred_names: HashMap::new(),
            descriptions: HashMap::new(),
            see_refs: Vec::new(),
        }
    }

    /// Add a preferred name for a language
    pub fn add_preferred_name(&mut self, lang: String, name: String) {
        self.preferred_names.insert(lang, name);
    }

    /// Add a description for a language
    pub fn add_description(&mut self, lang: String, description: String) {
        self.descriptions.insert(lang, description);
    }

    /// Add a see reference
    pub fn add_see_ref(&mut self, url: String) {
        self.see_refs.push(url);
    }

    /// Get preferred name for a language, falling back to English or first available
    pub fn get_preferred_name(&self, lang: &str) -> Option<&str> {
        self.preferred_names
            .get(lang)
            .or_else(|| self.preferred_names.get("en"))
            .or_else(|| self.preferred_names.values().next())
            .map(|s| s.as_str())
    }

    /// Get description for a language, falling back to English or first available
    pub fn get_description(&self, lang: &str) -> Option<&str> {
        self.descriptions
            .get(lang)
            .or_else(|| self.descriptions.get("en"))
            .or_else(|| self.descriptions.values().next())
            .map(|s| s.as_str())
    }
}

/// Trait for all SAMM model elements
pub trait ModelElement {
    /// Get the URN of this element
    fn urn(&self) -> &str;

    /// Get the element metadata
    fn metadata(&self) -> &ElementMetadata;

    /// Get the simple name from the URN
    fn name(&self) -> String {
        self.urn()
            .split('#')
            .next_back()
            .unwrap_or(self.urn())
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_metadata() {
        let mut meta = ElementMetadata::new("urn:samm:org.example:1.0.0#TestElement".to_string());

        meta.add_preferred_name("en".to_string(), "Test Element".to_string());
        meta.add_preferred_name("de".to_string(), "Test Element".to_string());

        meta.add_description("en".to_string(), "A test element".to_string());

        assert_eq!(meta.get_preferred_name("en"), Some("Test Element"));
        assert_eq!(meta.get_preferred_name("de"), Some("Test Element"));
        assert_eq!(meta.get_preferred_name("fr"), Some("Test Element")); // Falls back to en

        assert_eq!(meta.get_description("en"), Some("A test element"));
    }
}
