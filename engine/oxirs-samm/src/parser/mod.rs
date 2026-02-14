//! SAMM Model Parser
//!
//! This module provides functionality to parse SAMM models from Turtle/RDF format.

pub mod error_recovery;
pub mod incremental;
mod resolver;
mod streaming;
mod ttl_parser;

pub use error_recovery::{ErrorRecoveryStrategy, RecoveryAction, RecoveryContext};
pub use incremental::{IncrementalParser, ParseEvent, ParseState};
pub use resolver::ModelResolver;
pub use streaming::StreamingParser;
pub use ttl_parser::SammTurtleParser;

use crate::error::{Result, SammError};
use crate::metamodel::Aspect;
use std::path::Path;

/// Parse a SAMM Aspect model from a Turtle file
///
/// # Arguments
///
/// * `path` - Path to the Turtle file containing the Aspect model
///
/// # Returns
///
/// The parsed Aspect model
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_samm::parser::parse_aspect_model;
/// use oxirs_samm::metamodel::ModelElement;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let aspect = parse_aspect_model("path/to/AspectModel.ttl").await?;
/// println!("Parsed aspect: {}", aspect.name());
/// # Ok(())
/// # }
/// ```
pub async fn parse_aspect_model<P: AsRef<Path>>(path: P) -> Result<Aspect> {
    let mut parser = SammTurtleParser::new();
    parser.parse_file(path.as_ref()).await
}

/// Parse a SAMM model from a Turtle string
///
/// # Arguments
///
/// * `content` - Turtle/RDF content as a string
/// * `base_uri` - Base URI for resolving relative URIs
///
/// # Returns
///
/// The parsed Aspect model
pub async fn parse_aspect_from_string(content: &str, base_uri: &str) -> Result<Aspect> {
    let mut parser = SammTurtleParser::new();
    parser.parse_string(content, base_uri).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::ModelElement;

    #[tokio::test]
    async fn test_parse_from_string_basic() {
        let ttl_content = r#"
            @prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#> .
            @prefix : <urn:samm:org.example:1.0.0#> .

            :TestAspect a samm:Aspect ;
                samm:preferredName "Test Aspect"@en ;
                samm:description "A test aspect"@en ;
                samm:properties ( :testProperty ) .

            :testProperty a samm:Property ;
                samm:preferredName "Test Property"@en ;
                samm:characteristic :TestCharacteristic .

            :TestCharacteristic a samm:Characteristic ;
                samm:dataType <http://www.w3.org/2001/XMLSchema#string> .
        "#;

        let result = parse_aspect_from_string(ttl_content, "urn:samm:org.example:1.0.0#").await;
        assert!(
            result.is_ok(),
            "Failed to parse basic Turtle: {:?}",
            result.err()
        );

        let aspect = result.expect("result should be Ok");
        assert_eq!(aspect.name(), "TestAspect");
        assert_eq!(aspect.properties().len(), 1);
        assert_eq!(aspect.properties()[0].name(), "testProperty");
    }

    #[tokio::test]
    async fn test_parse_from_string_invalid_ttl() {
        let invalid_ttl = r#"
            @prefix samm: <invalid syntax here
        "#;

        let result = parse_aspect_from_string(invalid_ttl, "urn:samm:org.example:1.0.0#").await;
        assert!(result.is_err(), "Expected error for invalid Turtle syntax");
    }

    #[tokio::test]
    async fn test_parse_from_string_no_aspect() {
        let ttl_content = r#"
            @prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#> .
            @prefix : <urn:samm:org.example:1.0.0#> .

            :TestProperty a samm:Property ;
                samm:preferredName "Test Property"@en .
        "#;

        let result = parse_aspect_from_string(ttl_content, "urn:samm:org.example:1.0.0#").await;
        assert!(result.is_err(), "Expected error when no Aspect is defined");

        if let Err(err) = result {
            let err_str = format!("{:?}", err);
            assert!(
                err_str.contains("No Aspect found"),
                "Expected 'No Aspect found' error"
            );
        }
    }

    #[tokio::test]
    async fn test_parse_aspect_with_multiple_properties() {
        let ttl_content = r#"
            @prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#> .
            @prefix : <urn:samm:org.example:1.0.0#> .

            :MultiPropAspect a samm:Aspect ;
                samm:preferredName "Multi Property Aspect"@en ;
                samm:properties ( :prop1 :prop2 :prop3 ) .

            :prop1 a samm:Property ;
                samm:characteristic :StringChar .

            :prop2 a samm:Property ;
                samm:characteristic :IntChar .

            :prop3 a samm:Property ;
                samm:characteristic :BoolChar .

            :StringChar a samm:Characteristic ;
                samm:dataType <http://www.w3.org/2001/XMLSchema#string> .

            :IntChar a samm:Characteristic ;
                samm:dataType <http://www.w3.org/2001/XMLSchema#int> .

            :BoolChar a samm:Characteristic ;
                samm:dataType <http://www.w3.org/2001/XMLSchema#boolean> .
        "#;

        let result = parse_aspect_from_string(ttl_content, "urn:samm:org.example:1.0.0#").await;
        assert!(
            result.is_ok(),
            "Failed to parse multi-property aspect: {:?}",
            result.err()
        );

        let aspect = result.expect("result should be Ok");
        assert_eq!(aspect.properties().len(), 3);
        assert_eq!(aspect.properties()[0].name(), "prop1");
        assert_eq!(aspect.properties()[1].name(), "prop2");
        assert_eq!(aspect.properties()[2].name(), "prop3");
    }

    #[tokio::test]
    async fn test_parse_aspect_with_operations() {
        let ttl_content = r#"
            @prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#> .
            @prefix : <urn:samm:org.example:1.0.0#> .

            :OperationAspect a samm:Aspect ;
                samm:preferredName "Operation Aspect"@en ;
                samm:operations ( :testOperation ) .

            :testOperation a samm:Operation ;
                samm:preferredName "Test Operation"@en .
        "#;

        let result = parse_aspect_from_string(ttl_content, "urn:samm:org.example:1.0.0#").await;
        assert!(
            result.is_ok(),
            "Failed to parse aspect with operations: {:?}",
            result.err()
        );

        let aspect = result.expect("result should be Ok");
        assert_eq!(aspect.operations().len(), 1);
        assert_eq!(aspect.operations()[0].name(), "testOperation");
    }

    #[tokio::test]
    async fn test_model_resolver_creation() {
        let resolver = ModelResolver::new();
        let stats = resolver.cache_stats();
        assert_eq!(stats.content_cache_size, 0);
        assert_eq!(stats.path_cache_size, 0);
    }

    #[tokio::test]
    async fn test_model_resolver_add_models_root() {
        let mut resolver = ModelResolver::new();
        let test_path = std::env::temp_dir();
        resolver.add_models_root(test_path.clone());

        // Verify the path was added by checking it doesn't panic
        // (actual resolution would require real files)
    }
}
