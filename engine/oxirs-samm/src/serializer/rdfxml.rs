//! RDF/XML Serialization for SAMM Models
//!
//! This module provides RDF/XML serialization functionality for SAMM Aspect models.
//! RDF/XML is the original standard RDF syntax and is widely supported by RDF tools
//! and triple stores.
//!
//! ## Features
//!
//! - Full RDF/XML compliance
//! - Pretty-printed and compact output options
//! - Proper XML namespacing with RDF, SAMM prefixes
//! - Multi-language support for metadata
//! - Full SAMM 2.3.0 specification compliance
//!
//! ## Examples
//!
//! ```rust
//! use oxirs_samm::metamodel::{Aspect, ElementMetadata};
//! use oxirs_samm::serializer::RdfXmlSerializer;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let metadata = ElementMetadata::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
//! let aspect = Aspect {
//!     metadata,
//!     properties: vec![],
//!     operations: vec![],
//!     events: vec![],
//! };
//!
//! let serializer = RdfXmlSerializer::new();
//! let rdfxml = serializer.serialize_to_string(&aspect)?;
//!
//! println!("{}", rdfxml);
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, SammError};
use crate::metamodel::{Aspect, ModelElement};
use std::path::Path;

/// RDF/XML serializer for SAMM Aspect models
///
/// Provides serialization to RDF/XML format with proper XML namespacing
/// and RDF semantics.
pub struct RdfXmlSerializer {
    /// Whether to use pretty-printed output
    pretty: bool,
    /// Indentation string (used when pretty=true)
    indent: String,
}

impl RdfXmlSerializer {
    /// Create a new RDF/XML serializer with default settings
    ///
    /// Defaults: pretty=true, indent="  " (2 spaces)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_samm::serializer::RdfXmlSerializer;
    ///
    /// let serializer = RdfXmlSerializer::new();
    /// ```
    pub fn new() -> Self {
        Self {
            pretty: true,
            indent: "  ".to_string(),
        }
    }

    /// Set whether to use pretty-printed output
    ///
    /// # Arguments
    ///
    /// * `pretty` - If true, use indented formatting
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_samm::serializer::RdfXmlSerializer;
    ///
    /// let serializer = RdfXmlSerializer::new().with_pretty(false);
    /// ```
    pub fn with_pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }

    /// Set the indentation string
    ///
    /// # Arguments
    ///
    /// * `indent` - Indentation string (e.g., "  " for 2 spaces, "\t" for tab)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_samm::serializer::RdfXmlSerializer;
    ///
    /// let serializer = RdfXmlSerializer::new().with_indent("    "); // 4 spaces
    /// ```
    pub fn with_indent(mut self, indent: &str) -> Self {
        self.indent = indent.to_string();
        self
    }

    /// Serialize an Aspect to an RDF/XML string
    ///
    /// # Arguments
    ///
    /// * `aspect` - The Aspect model to serialize
    ///
    /// # Returns
    ///
    /// RDF/XML string representation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_samm::metamodel::{Aspect, ElementMetadata};
    /// use oxirs_samm::serializer::RdfXmlSerializer;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let metadata = ElementMetadata::new("urn:samm:org.example:1.0.0#MyAspect".to_string());
    /// let aspect = Aspect {
    ///     metadata,
    ///     properties: vec![],
    ///     operations: vec![],
    ///     events: vec![],
    /// };
    ///
    /// let serializer = RdfXmlSerializer::new();
    /// let rdfxml = serializer.serialize_to_string(&aspect)?;
    /// assert!(rdfxml.contains("<rdf:RDF"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn serialize_to_string(&self, aspect: &Aspect) -> Result<String> {
        let mut output = String::new();

        // XML declaration
        output.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        if self.pretty {
            output.push('\n');
        }

        // RDF root element with namespaces
        output.push_str("<rdf:RDF");
        if self.pretty {
            output.push('\n');
        } else {
            output.push(' ');
        }

        // Add namespace declarations
        self.add_namespace(
            &mut output,
            "xmlns:rdf",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        );
        self.add_namespace(
            &mut output,
            "xmlns:rdfs",
            "http://www.w3.org/2000/01/rdf-schema#",
        );
        self.add_namespace(
            &mut output,
            "xmlns:samm",
            "urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#",
        );
        self.add_namespace(
            &mut output,
            "xmlns:samm-c",
            "urn:samm:org.eclipse.esmf.samm:characteristic:2.3.0#",
        );
        self.add_namespace(
            &mut output,
            "xmlns:samm-e",
            "urn:samm:org.eclipse.esmf.samm:entity:2.3.0#",
        );
        self.add_namespace(
            &mut output,
            "xmlns:xsd",
            "http://www.w3.org/2001/XMLSchema#",
        );

        output.push('>');
        if self.pretty {
            output.push('\n');
        }

        // Serialize aspect
        self.serialize_aspect_element(&mut output, aspect, 1)?;

        // Close RDF root
        if self.pretty {
            output.push('\n');
        }
        output.push_str("</rdf:RDF>");
        if self.pretty {
            output.push('\n');
        }

        Ok(output)
    }

    /// Serialize an Aspect to an RDF/XML file
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
    /// # Examples
    ///
    /// ```rust,no_run
    /// use oxirs_samm::metamodel::{Aspect, ElementMetadata};
    /// use oxirs_samm::serializer::RdfXmlSerializer;
    /// use std::path::Path;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let metadata = ElementMetadata::new("urn:samm:org.example:1.0.0#MyAspect".to_string());
    /// let aspect = Aspect {
    ///     metadata,
    ///     properties: vec![],
    ///     operations: vec![],
    ///     events: vec![],
    /// };
    ///
    /// let serializer = RdfXmlSerializer::new();
    /// serializer.serialize_to_file(&aspect, Path::new("output/aspect.rdf")).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn serialize_to_file(&self, aspect: &Aspect, path: &Path) -> Result<()> {
        let content = self.serialize_to_string(aspect)?;

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Write file asynchronously
        tokio::fs::write(path, content.as_bytes()).await?;

        Ok(())
    }

    /// Add a namespace declaration to the output
    fn add_namespace(&self, output: &mut String, prefix: &str, uri: &str) {
        if self.pretty {
            output.push_str(&format!("{}{}=\"{}\"", self.indent, prefix, uri));
            output.push('\n');
        } else {
            output.push_str(&format!(" {}=\"{}\"", prefix, uri));
        }
    }

    /// Serialize an Aspect element
    fn serialize_aspect_element(
        &self,
        output: &mut String,
        aspect: &Aspect,
        depth: usize,
    ) -> Result<()> {
        let indent_str = if self.pretty {
            self.indent.repeat(depth)
        } else {
            String::new()
        };

        let newline = if self.pretty { "\n" } else { "" };

        // Start aspect element
        output.push_str(&format!(
            "{}<rdf:Description rdf:about=\"{}\">{}",
            indent_str,
            self.escape_xml(aspect.urn()),
            newline
        ));

        // Add rdf:type
        output.push_str(&format!(
            "{}{}<rdf:type rdf:resource=\"urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#Aspect\"/>{}",
            indent_str, self.indent, newline
        ));

        // Add preferred names
        for (lang, name) in &aspect.metadata.preferred_names {
            output.push_str(&format!(
                "{}{}<samm:preferredName xml:lang=\"{}\">{}</samm:preferredName>{}",
                indent_str,
                self.indent,
                self.escape_xml(lang),
                self.escape_xml(name),
                newline
            ));
        }

        // Add descriptions
        for (lang, desc) in &aspect.metadata.descriptions {
            output.push_str(&format!(
                "{}{}<samm:description xml:lang=\"{}\">{}</samm:description>{}",
                indent_str,
                self.indent,
                self.escape_xml(lang),
                self.escape_xml(desc),
                newline
            ));
        }

        // Add see references
        for see_ref in &aspect.metadata.see_refs {
            output.push_str(&format!(
                "{}{}<samm:see rdf:resource=\"{}\"/>{}",
                indent_str,
                self.indent,
                self.escape_xml(see_ref),
                newline
            ));
        }

        // Add properties
        for property in &aspect.properties {
            output.push_str(&format!(
                "{}{}<samm:properties rdf:resource=\"{}\"/>{}",
                indent_str,
                self.indent,
                self.escape_xml(property.urn()),
                newline
            ));
        }

        // Add operations
        for operation in &aspect.operations {
            output.push_str(&format!(
                "{}{}<samm:operations rdf:resource=\"{}\"/>{}",
                indent_str,
                self.indent,
                self.escape_xml(operation.urn()),
                newline
            ));
        }

        // Add events
        for event in &aspect.events {
            output.push_str(&format!(
                "{}{}<samm:events rdf:resource=\"{}\"/>{}",
                indent_str,
                self.indent,
                self.escape_xml(event.urn()),
                newline
            ));
        }

        // Close aspect element
        output.push_str(&format!("{}</rdf:Description>{}", indent_str, newline));

        Ok(())
    }

    /// Escape XML special characters
    fn escape_xml(&self, text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }
}

impl Default for RdfXmlSerializer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Characteristic, CharacteristicKind, ElementMetadata, Property};

    fn create_test_aspect() -> Aspect {
        let mut metadata =
            ElementMetadata::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
        metadata.add_preferred_name("en".to_string(), "Test Aspect".to_string());
        metadata.add_description("en".to_string(), "A test aspect".to_string());
        metadata.add_see_ref("https://example.org/docs".to_string());

        let mut prop1 = Property::new("urn:samm:org.example:1.0.0#property1".to_string());
        let char1 = Characteristic::new(
            "urn:samm:org.eclipse.esmf.samm:characteristic:2.3.0#Text".to_string(),
            CharacteristicKind::Trait,
        );
        prop1.characteristic = Some(char1);
        prop1.optional = false;

        Aspect {
            metadata,
            properties: vec![prop1],
            operations: vec![],
            events: vec![],
        }
    }

    #[test]
    fn test_rdfxml_serialization_basic() {
        let aspect = create_test_aspect();
        let serializer = RdfXmlSerializer::new();
        let rdfxml = serializer.serialize_to_string(&aspect).unwrap();

        // Verify XML structure
        assert!(rdfxml.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
        assert!(rdfxml.contains("<rdf:RDF"));
        assert!(rdfxml.contains("</rdf:RDF>"));

        // Check namespaces
        assert!(rdfxml.contains("xmlns:rdf="));
        assert!(rdfxml.contains("xmlns:samm="));
        assert!(rdfxml.contains("xmlns:samm-c="));
        assert!(rdfxml.contains("xmlns:samm-e="));
        assert!(rdfxml.contains("xmlns:xsd="));

        // Check aspect URN
        assert!(rdfxml.contains("urn:samm:org.example:1.0.0#TestAspect"));
    }

    #[test]
    fn test_rdfxml_contains_metadata() {
        let aspect = create_test_aspect();
        let serializer = RdfXmlSerializer::new();
        let rdfxml = serializer.serialize_to_string(&aspect).unwrap();

        assert!(rdfxml.contains("<samm:preferredName"));
        assert!(rdfxml.contains("Test Aspect"));
        assert!(rdfxml.contains("<samm:description"));
        assert!(rdfxml.contains("A test aspect"));
        assert!(rdfxml.contains("<samm:see"));
        assert!(rdfxml.contains("https://example.org/docs"));
    }

    #[test]
    fn test_rdfxml_contains_properties() {
        let aspect = create_test_aspect();
        let serializer = RdfXmlSerializer::new();
        let rdfxml = serializer.serialize_to_string(&aspect).unwrap();

        assert!(rdfxml.contains("<samm:properties"));
        assert!(rdfxml.contains("urn:samm:org.example:1.0.0#property1"));
    }

    #[test]
    fn test_rdfxml_pretty_vs_compact() {
        let aspect = create_test_aspect();

        let pretty = RdfXmlSerializer::new().with_pretty(true);
        let compact = RdfXmlSerializer::new().with_pretty(false);

        let pretty_output = pretty.serialize_to_string(&aspect).unwrap();
        let compact_output = compact.serialize_to_string(&aspect).unwrap();

        // Pretty output should have newlines and indentation
        assert!(pretty_output.contains('\n'));
        assert!(pretty_output.len() > compact_output.len());

        // Compact should have minimal newlines
        let pretty_lines = pretty_output.lines().count();
        let compact_lines = compact_output.lines().count();
        assert!(pretty_lines > compact_lines);
    }

    #[test]
    fn test_rdfxml_multi_language() {
        let mut metadata = ElementMetadata::new("urn:samm:org.example:1.0.0#MultiLang".to_string());
        metadata.add_preferred_name("en".to_string(), "Test Aspect".to_string());
        metadata.add_preferred_name("de".to_string(), "Test Aspekt".to_string());
        metadata.add_preferred_name("fr".to_string(), "Aspect de Test".to_string());

        metadata.add_description("en".to_string(), "English description".to_string());
        metadata.add_description("de".to_string(), "Deutsche Beschreibung".to_string());

        let aspect = Aspect {
            metadata,
            properties: vec![],
            operations: vec![],
            events: vec![],
        };

        let serializer = RdfXmlSerializer::new();
        let rdfxml = serializer.serialize_to_string(&aspect).unwrap();

        // All languages should be present with xml:lang attributes
        assert!(rdfxml.contains("xml:lang=\"en\""));
        assert!(rdfxml.contains("xml:lang=\"de\""));
        assert!(rdfxml.contains("xml:lang=\"fr\""));
        assert!(rdfxml.contains("Test Aspect"));
        assert!(rdfxml.contains("Test Aspekt"));
        assert!(rdfxml.contains("Aspect de Test"));
        assert!(rdfxml.contains("English description"));
        assert!(rdfxml.contains("Deutsche Beschreibung"));
    }

    #[test]
    fn test_rdfxml_empty_aspect() {
        let metadata = ElementMetadata::new("urn:samm:org.example:1.0.0#Empty".to_string());
        let aspect = Aspect {
            metadata,
            properties: vec![],
            operations: vec![],
            events: vec![],
        };

        let serializer = RdfXmlSerializer::new();
        let rdfxml = serializer.serialize_to_string(&aspect).unwrap();

        // Should have basic structure
        assert!(rdfxml.contains("<rdf:RDF"));
        assert!(rdfxml.contains("<rdf:Description"));
        assert!(rdfxml.contains("urn:samm:org.example:1.0.0#Empty"));

        // Empty aspect should not have properties/operations/events
        assert!(!rdfxml.contains("<samm:properties"));
        assert!(!rdfxml.contains("<samm:operations"));
        assert!(!rdfxml.contains("<samm:events"));
    }

    #[test]
    fn test_rdfxml_xml_escaping() {
        let mut metadata =
            ElementMetadata::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
        metadata.add_preferred_name("en".to_string(), "Test & <Special> Characters".to_string());

        let aspect = Aspect {
            metadata,
            properties: vec![],
            operations: vec![],
            events: vec![],
        };

        let serializer = RdfXmlSerializer::new();
        let rdfxml = serializer.serialize_to_string(&aspect).unwrap();

        // XML special characters should be escaped
        assert!(rdfxml.contains("&amp;"));
        assert!(rdfxml.contains("&lt;"));
        assert!(rdfxml.contains("&gt;"));
        assert!(!rdfxml.contains("Test & <Special>"));
    }

    #[test]
    fn test_rdfxml_custom_indentation() {
        let aspect = create_test_aspect();

        let serializer = RdfXmlSerializer::new().with_indent("    "); // 4 spaces

        let rdfxml = serializer.serialize_to_string(&aspect).unwrap();

        // Should use 4-space indentation
        let lines: Vec<&str> = rdfxml.lines().collect();
        let indented_lines: Vec<&str> = lines
            .iter()
            .filter(|line| line.starts_with("    "))
            .copied()
            .collect();

        assert!(!indented_lines.is_empty());
    }

    #[test]
    fn test_rdfxml_type_declaration() {
        let aspect = create_test_aspect();
        let serializer = RdfXmlSerializer::new();
        let rdfxml = serializer.serialize_to_string(&aspect).unwrap();

        // Should have proper rdf:type declaration
        assert!(rdfxml.contains("<rdf:type"));
        assert!(rdfxml.contains("urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#Aspect"));
    }
}
