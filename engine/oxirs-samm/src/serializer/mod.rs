//! SAMM Model Serialization
//!
//! This module provides functionality to serialize SAMM models to Turtle/RDF format.

mod turtle;

pub use turtle::TurtleSerializer;

use crate::error::Result;
use crate::metamodel::Aspect;
use std::path::Path;

/// Serialize a SAMM Aspect model to a Turtle file
///
/// # Arguments
///
/// * `aspect` - The Aspect model to serialize
/// * `path` - Output file path
///
/// # Returns
///
/// Result indicating success or failure
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_samm::metamodel::{Aspect, ElementMetadata};
/// use oxirs_samm::serializer::serialize_aspect_to_file;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let metadata = ElementMetadata::new("urn:example:aspect#MyAspect".to_string());
/// let aspect = Aspect {
///     metadata,
///     properties: vec![],
///     operations: vec![],
///     events: vec![],
/// };
///
/// serialize_aspect_to_file(&aspect, "output/MyAspect.ttl").await?;
/// # Ok(())
/// # }
/// ```
pub async fn serialize_aspect_to_file<P: AsRef<Path>>(aspect: &Aspect, path: P) -> Result<()> {
    let serializer = TurtleSerializer::new();
    serializer.serialize_to_file(aspect, path.as_ref()).await
}

/// Serialize a SAMM Aspect model to a Turtle string
///
/// # Arguments
///
/// * `aspect` - The Aspect model to serialize
///
/// # Returns
///
/// Turtle/RDF content as a string
pub fn serialize_aspect_to_string(aspect: &Aspect) -> Result<String> {
    let serializer = TurtleSerializer::new();
    serializer.serialize_to_string(aspect)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::ElementMetadata;

    #[test]
    fn test_serialize_minimal_aspect() {
        let metadata = ElementMetadata::new("urn:example:aspect#TestAspect".to_string());
        let aspect = Aspect {
            metadata,
            properties: vec![],
            operations: vec![],
            events: vec![],
        };

        let result = serialize_aspect_to_string(&aspect);
        assert!(result.is_ok());

        let ttl = result.unwrap();
        assert!(ttl.contains("@prefix samm:"));
        assert!(ttl.contains("urn:example:aspect#TestAspect"));
        assert!(ttl.contains("a samm:Aspect"));
    }
}
