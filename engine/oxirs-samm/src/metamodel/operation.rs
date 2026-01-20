//! Operation and Event Model Elements
//!
//! Operations are functions that can be performed on Aspects.
//! Events are occurrences that can be emitted by Aspects.

use super::{ElementMetadata, ModelElement, Property};
use serde::{Deserialize, Serialize};

/// An Operation in the SAMM meta model
///
/// Operations represent functions that can be performed on an Aspect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    /// Element metadata (URN, names, descriptions)
    pub metadata: ElementMetadata,

    /// Input parameters for this operation
    pub input: Vec<Property>,

    /// Output of this operation
    pub output: Option<Property>,
}

impl Operation {
    /// Create a new Operation
    pub fn new(urn: String) -> Self {
        Self {
            metadata: ElementMetadata::new(urn),
            input: Vec::new(),
            output: None,
        }
    }

    /// Get the input parameters
    pub fn input(&self) -> &[Property] {
        &self.input
    }

    /// Add an input parameter
    pub fn add_input(&mut self, property: Property) {
        self.input.push(property);
    }

    /// Set the output
    pub fn with_output(mut self, property: Property) -> Self {
        self.output = Some(property);
        self
    }

    /// Get the output
    pub fn output(&self) -> Option<&Property> {
        self.output.as_ref()
    }
}

impl ModelElement for Operation {
    fn urn(&self) -> &str {
        &self.metadata.urn
    }

    fn metadata(&self) -> &ElementMetadata {
        &self.metadata
    }
}

/// An Event in the SAMM meta model
///
/// Events represent occurrences that can be emitted by an Aspect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Element metadata (URN, names, descriptions)
    pub metadata: ElementMetadata,

    /// Parameters of this event
    pub parameters: Vec<Property>,
}

impl Event {
    /// Create a new Event
    pub fn new(urn: String) -> Self {
        Self {
            metadata: ElementMetadata::new(urn),
            parameters: Vec::new(),
        }
    }

    /// Get the parameters
    pub fn parameters(&self) -> &[Property] {
        &self.parameters
    }

    /// Add a parameter
    pub fn add_parameter(&mut self, property: Property) {
        self.parameters.push(property);
    }
}

impl ModelElement for Event {
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
    fn test_operation_creation() {
        let operation = Operation::new("urn:samm:org.example:1.0.0#testOperation".to_string());

        assert_eq!(operation.name(), "testOperation");
        assert_eq!(operation.input().len(), 0);
        assert!(operation.output().is_none());
    }

    #[test]
    fn test_operation_with_output() {
        let output = Property::new("urn:samm:org.example:1.0.0#output".to_string());
        let operation =
            Operation::new("urn:samm:org.example:1.0.0#testOp".to_string()).with_output(output);

        assert!(operation.output().is_some());
    }

    #[test]
    fn test_event_creation() {
        let event = Event::new("urn:samm:org.example:1.0.0#testEvent".to_string());

        assert_eq!(event.name(), "testEvent");
        assert_eq!(event.parameters().len(), 0);
    }
}
