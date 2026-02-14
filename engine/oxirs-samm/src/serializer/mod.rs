//! SAMM Model Serialization
//!
//! This module provides functionality to serialize SAMM models to various RDF formats.
//!
//! ## Supported Formats
//!
//! - **Turtle** - RDF Turtle format (.ttl)
//! - **JSON-LD** - JSON-based linked data format (.jsonld)
//! - **RDF/XML** - XML-based RDF format (.rdf, .xml)
//!
//! ## Examples
//!
//! ```rust
//! use oxirs_samm::metamodel::{Aspect, ElementMetadata};
//! use oxirs_samm::serializer::{TurtleSerializer, JsonLdSerializer, RdfXmlSerializer};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let metadata = ElementMetadata::new("urn:samm:org.example:1.0.0#MyAspect".to_string());
//! let aspect = Aspect {
//!     metadata,
//!     properties: vec![],
//!     operations: vec![],
//!     events: vec![],
//! };
//!
//! // Serialize to Turtle
//! let turtle = TurtleSerializer::new();
//! let ttl = turtle.serialize_to_string(&aspect)?;
//!
//! // Serialize to JSON-LD
//! let jsonld = JsonLdSerializer::new();
//! let json = jsonld.serialize_to_string(&aspect)?;
//!
//! // Serialize to RDF/XML
//! let rdfxml = RdfXmlSerializer::new();
//! let xml = rdfxml.serialize_to_string(&aspect)?;
//! # Ok(())
//! # }
//! ```

mod jsonld;
mod rdfxml;
mod turtle;

pub use jsonld::JsonLdSerializer;
pub use rdfxml::RdfXmlSerializer;
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

/// Serialize a SAMM Aspect model to a JSON-LD file
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
/// use oxirs_samm::serializer::serialize_aspect_to_jsonld_file;
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
/// serialize_aspect_to_jsonld_file(&aspect, "output/MyAspect.jsonld").await?;
/// # Ok(())
/// # }
/// ```
pub async fn serialize_aspect_to_jsonld_file<P: AsRef<Path>>(
    aspect: &Aspect,
    path: P,
) -> Result<()> {
    let serializer = JsonLdSerializer::new();
    serializer.serialize_to_file(aspect, path.as_ref()).await
}

/// Serialize a SAMM Aspect model to a JSON-LD string
///
/// # Arguments
///
/// * `aspect` - The Aspect model to serialize
///
/// # Returns
///
/// JSON-LD content as a string
///
/// # Example
///
/// ```rust
/// use oxirs_samm::metamodel::{Aspect, ElementMetadata};
/// use oxirs_samm::serializer::serialize_aspect_to_jsonld_string;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let metadata = ElementMetadata::new("urn:example:aspect#MyAspect".to_string());
/// let aspect = Aspect {
///     metadata,
///     properties: vec![],
///     operations: vec![],
///     events: vec![],
/// };
///
/// let jsonld = serialize_aspect_to_jsonld_string(&aspect)?;
/// assert!(jsonld.contains("@context"));
/// # Ok(())
/// # }
/// ```
pub fn serialize_aspect_to_jsonld_string(aspect: &Aspect) -> Result<String> {
    let serializer = JsonLdSerializer::new();
    serializer.serialize_to_string(aspect)
}

/// Serialize a SAMM Aspect model to an RDF/XML file
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
/// use oxirs_samm::serializer::serialize_aspect_to_rdfxml_file;
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
/// serialize_aspect_to_rdfxml_file(&aspect, "output/MyAspect.rdf").await?;
/// # Ok(())
/// # }
/// ```
pub async fn serialize_aspect_to_rdfxml_file<P: AsRef<Path>>(
    aspect: &Aspect,
    path: P,
) -> Result<()> {
    let serializer = RdfXmlSerializer::new();
    serializer.serialize_to_file(aspect, path.as_ref()).await
}

/// Serialize a SAMM Aspect model to an RDF/XML string
///
/// # Arguments
///
/// * `aspect` - The Aspect model to serialize
///
/// # Returns
///
/// RDF/XML content as a string
///
/// # Example
///
/// ```rust
/// use oxirs_samm::metamodel::{Aspect, ElementMetadata};
/// use oxirs_samm::serializer::serialize_aspect_to_rdfxml_string;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let metadata = ElementMetadata::new("urn:example:aspect#MyAspect".to_string());
/// let aspect = Aspect {
///     metadata,
///     properties: vec![],
///     operations: vec![],
///     events: vec![],
/// };
///
/// let rdfxml = serialize_aspect_to_rdfxml_string(&aspect)?;
/// assert!(rdfxml.contains("<rdf:RDF"));
/// # Ok(())
/// # }
/// ```
pub fn serialize_aspect_to_rdfxml_string(aspect: &Aspect) -> Result<String> {
    let serializer = RdfXmlSerializer::new();
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

        let ttl = result.expect("result should be Ok");
        assert!(ttl.contains("@prefix samm:"));
        assert!(ttl.contains("urn:example:aspect#TestAspect"));
        assert!(ttl.contains("a samm:Aspect"));
    }
}
