//! Aspect Model Element
//!
//! An Aspect is the root element describing a specific aspect of a digital twin.

use super::{ElementMetadata, ModelElement, Operation, Property};
use serde::{Deserialize, Serialize};

/// An Aspect in the SAMM meta model
///
/// Aspects are the root elements of SAMM models and describe specific functionalities
/// or features of digital twins.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aspect {
    /// Element metadata (URN, names, descriptions)
    pub metadata: ElementMetadata,

    /// Properties of this Aspect
    pub properties: Vec<Property>,

    /// Operations that can be performed on this Aspect
    pub operations: Vec<Operation>,

    /// Events that can be emitted by this Aspect
    pub events: Vec<super::Event>,
}

impl Aspect {
    /// Create a new Aspect
    pub fn new(urn: String) -> Self {
        Self {
            metadata: ElementMetadata::new(urn),
            properties: Vec::new(),
            operations: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Get the properties of this Aspect
    pub fn properties(&self) -> &[Property] {
        &self.properties
    }

    /// Add a property to this Aspect
    pub fn add_property(&mut self, property: Property) {
        self.properties.push(property);
    }

    /// Get the operations of this Aspect
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// Add an operation to this Aspect
    pub fn add_operation(&mut self, operation: Operation) {
        self.operations.push(operation);
    }

    /// Get the events of this Aspect
    pub fn events(&self) -> &[super::Event] {
        &self.events
    }

    /// Add an event to this Aspect
    pub fn add_event(&mut self, event: super::Event) {
        self.events.push(event);
    }
}

impl ModelElement for Aspect {
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
    fn test_aspect_creation() {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
        aspect
            .metadata
            .add_preferred_name("en".to_string(), "Test Aspect".to_string());

        assert_eq!(aspect.name(), "TestAspect");
        assert_eq!(aspect.properties().len(), 0);
        assert_eq!(aspect.operations().len(), 0);
    }
}
