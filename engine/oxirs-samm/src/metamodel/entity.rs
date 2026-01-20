//! Entity Model Elements
//!
//! Entities represent complex data structures with multiple properties.

use super::{ElementMetadata, ModelElement, Property};
use serde::{Deserialize, Serialize};

/// An Entity in the SAMM meta model
///
/// Entities are complex data structures that bundle multiple properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Element metadata (URN, names, descriptions)
    pub metadata: ElementMetadata,

    /// Properties of this Entity
    pub properties: Vec<Property>,

    /// If this entity extends another entity
    pub extends: Option<String>,

    /// Whether this entity is abstract (cannot be instantiated directly)
    pub is_abstract: bool,
}

impl Entity {
    /// Create a new Entity
    pub fn new(urn: String) -> Self {
        Self {
            metadata: ElementMetadata::new(urn),
            properties: Vec::new(),
            extends: None,
            is_abstract: false,
        }
    }

    /// Get the properties of this Entity
    pub fn properties(&self) -> &[Property] {
        &self.properties
    }

    /// Add a property to this Entity
    pub fn add_property(&mut self, property: Property) {
        self.properties.push(property);
    }

    /// Mark this entity as abstract
    pub fn as_abstract(mut self) -> Self {
        self.is_abstract = true;
        self
    }

    /// Set the entity this extends
    pub fn extends(mut self, entity_urn: String) -> Self {
        self.extends = Some(entity_urn);
        self
    }
}

impl ModelElement for Entity {
    fn urn(&self) -> &str {
        &self.metadata.urn
    }

    fn metadata(&self) -> &ElementMetadata {
        &self.metadata
    }
}

/// A ComplexType is similar to an Entity but used in different contexts
pub type ComplexType = Entity;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_creation() {
        let entity = Entity::new("urn:samm:org.example:1.0.0#TestEntity".to_string());

        assert_eq!(entity.name(), "TestEntity");
        assert_eq!(entity.properties().len(), 0);
        assert!(!entity.is_abstract);
    }

    #[test]
    fn test_abstract_entity() {
        let entity =
            Entity::new("urn:samm:org.example:1.0.0#AbstractEntity".to_string()).as_abstract();

        assert!(entity.is_abstract);
    }

    #[test]
    fn test_entity_inheritance() {
        let entity = Entity::new("urn:samm:org.example:1.0.0#DerivedEntity".to_string())
            .extends("urn:samm:org.example:1.0.0#BaseEntity".to_string());

        assert_eq!(
            entity.extends,
            Some("urn:samm:org.example:1.0.0#BaseEntity".to_string())
        );
    }
}
