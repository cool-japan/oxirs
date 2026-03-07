//! JSON-LD Serialization for SAMM Models
//!
//! This module provides JSON-LD serialization functionality for SAMM Aspect models.
//! JSON-LD is a JSON-based format for linked data that is widely used in web applications
//! and provides excellent human readability while maintaining full RDF semantics.
//!
//! ## Features
//!
//! - Compact JSON-LD format with proper `@context`
//! - Multi-language support for metadata
//! - Full SAMM 2.3.0 specification compliance
//! - Pretty-printed and compact output options
//! - Proper URN handling with namespace prefixes
//!
//! ## Examples
//!
//! ```rust
//! use oxirs_samm::metamodel::{Aspect, ElementMetadata};
//! use oxirs_samm::serializer::JsonLdSerializer;
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
//! let serializer = JsonLdSerializer::new();
//! let jsonld = serializer.serialize_to_string(&aspect)?;
//!
//! println!("{}", jsonld);
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, SammError};
use crate::metamodel::{Aspect, CharacteristicKind, ModelElement};
use serde_json::{json, Map, Value};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// JSON-LD serializer for SAMM Aspect models
///
/// Provides serialization to JSON-LD format with proper RDF semantics
/// and SAMM namespace context.
pub struct JsonLdSerializer {
    /// Whether to use pretty-printed output
    pretty: bool,
    /// Whether to include full URNs or use prefixed names
    use_prefixes: bool,
}

impl JsonLdSerializer {
    /// Create a new JSON-LD serializer with default settings
    ///
    /// Defaults: pretty=true, use_prefixes=true
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_samm::serializer::JsonLdSerializer;
    ///
    /// let serializer = JsonLdSerializer::new();
    /// ```
    pub fn new() -> Self {
        Self {
            pretty: true,
            use_prefixes: true,
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
    /// use oxirs_samm::serializer::JsonLdSerializer;
    ///
    /// let serializer = JsonLdSerializer::new().with_pretty(false);
    /// ```
    pub fn with_pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }

    /// Set whether to use prefixed names or full URNs
    ///
    /// # Arguments
    ///
    /// * `use_prefixes` - If true, use samm:Aspect instead of full URN
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_samm::serializer::JsonLdSerializer;
    ///
    /// let serializer = JsonLdSerializer::new().with_prefixes(false);
    /// ```
    pub fn with_prefixes(mut self, use_prefixes: bool) -> Self {
        self.use_prefixes = use_prefixes;
        self
    }

    /// Serialize an Aspect to a JSON-LD string
    ///
    /// # Arguments
    ///
    /// * `aspect` - The Aspect model to serialize
    ///
    /// # Returns
    ///
    /// JSON-LD string representation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_samm::metamodel::{Aspect, ElementMetadata};
    /// use oxirs_samm::serializer::JsonLdSerializer;
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
    /// let serializer = JsonLdSerializer::new();
    /// let jsonld = serializer.serialize_to_string(&aspect)?;
    /// assert!(jsonld.contains("@context"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn serialize_to_string(&self, aspect: &Aspect) -> Result<String> {
        let json_value = self.serialize_aspect(aspect)?;

        let result = if self.pretty {
            serde_json::to_string_pretty(&json_value)
        } else {
            serde_json::to_string(&json_value)
        };

        result.map_err(|e| SammError::ParseError(format!("JSON-LD serialization error: {}", e)))
    }

    /// Serialize an Aspect to a JSON-LD file
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
    /// use oxirs_samm::serializer::JsonLdSerializer;
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
    /// let serializer = JsonLdSerializer::new();
    /// serializer.serialize_to_file(&aspect, Path::new("output/aspect.jsonld")).await?;
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

    /// Serialize an Aspect to a JSON-LD Value
    fn serialize_aspect(&self, aspect: &Aspect) -> Result<Value> {
        let mut obj = Map::new();

        // Add @context
        obj.insert("@context".to_string(), self.create_context());

        // Add @id (the URN of the aspect)
        obj.insert("@id".to_string(), Value::String(aspect.urn().to_string()));

        // Add @type
        let type_value = if self.use_prefixes {
            "samm:Aspect".to_string()
        } else {
            "http://www.w3.org/ns/shacl#NodeShape".to_string()
        };
        obj.insert("@type".to_string(), Value::String(type_value));

        // Add preferred names
        if !aspect.metadata.preferred_names.is_empty() {
            let mut names = Map::new();
            for (lang, name) in &aspect.metadata.preferred_names {
                names.insert(format!("@{}", lang), Value::String(name.clone()));
            }
            obj.insert("samm:preferredName".to_string(), Value::Object(names));
        }

        // Add descriptions
        if !aspect.metadata.descriptions.is_empty() {
            let mut descs = Map::new();
            for (lang, desc) in &aspect.metadata.descriptions {
                descs.insert(format!("@{}", lang), Value::String(desc.clone()));
            }
            obj.insert("samm:description".to_string(), Value::Object(descs));
        }

        // Add see references
        if !aspect.metadata.see_refs.is_empty() {
            let see_refs: Vec<Value> = aspect
                .metadata
                .see_refs
                .iter()
                .map(|url| Value::String(url.clone()))
                .collect();
            obj.insert("samm:see".to_string(), Value::Array(see_refs));
        }

        // Add properties
        if !aspect.properties.is_empty() {
            let properties: Vec<Value> = aspect
                .properties
                .iter()
                .map(|prop| {
                    let mut prop_obj = Map::new();
                    prop_obj.insert("@id".to_string(), Value::String(prop.urn().to_string()));

                    // Add property metadata if present
                    if !prop.metadata.preferred_names.is_empty() {
                        let mut names = Map::new();
                        for (lang, name) in &prop.metadata.preferred_names {
                            names.insert(format!("@{}", lang), Value::String(name.clone()));
                        }
                        prop_obj.insert("samm:preferredName".to_string(), Value::Object(names));
                    }

                    // Add characteristic reference
                    if let Some(ref char) = prop.characteristic {
                        let char_ref = if self.use_prefixes {
                            format!("samm-c:{}", char.name())
                        } else {
                            char.urn().to_string()
                        };
                        prop_obj
                            .insert("samm:characteristic".to_string(), json!({"@id": char_ref}));
                    }

                    // Add optional flag
                    if prop.optional {
                        prop_obj.insert("samm:optional".to_string(), Value::Bool(true));
                    }

                    Value::Object(prop_obj)
                })
                .collect();

            obj.insert("samm:properties".to_string(), Value::Array(properties));
        }

        // Add operations
        if !aspect.operations.is_empty() {
            let operations: Vec<Value> = aspect
                .operations
                .iter()
                .map(|op| {
                    let mut op_obj = Map::new();
                    op_obj.insert("@id".to_string(), Value::String(op.urn().to_string()));
                    Value::Object(op_obj)
                })
                .collect();

            obj.insert("samm:operations".to_string(), Value::Array(operations));
        }

        // Add events
        if !aspect.events.is_empty() {
            let events: Vec<Value> = aspect
                .events
                .iter()
                .map(|event| {
                    let mut event_obj = Map::new();
                    event_obj.insert("@id".to_string(), Value::String(event.urn().to_string()));
                    Value::Object(event_obj)
                })
                .collect();

            obj.insert("samm:events".to_string(), Value::Array(events));
        }

        Ok(Value::Object(obj))
    }

    /// Create the @context object for JSON-LD
    fn create_context(&self) -> Value {
        let mut context = Map::new();

        context.insert(
            "samm".to_string(),
            Value::String("urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#".to_string()),
        );

        context.insert(
            "samm-c".to_string(),
            Value::String("urn:samm:org.eclipse.esmf.samm:characteristic:2.3.0#".to_string()),
        );

        context.insert(
            "samm-e".to_string(),
            Value::String("urn:samm:org.eclipse.esmf.samm:entity:2.3.0#".to_string()),
        );

        context.insert(
            "xsd".to_string(),
            Value::String("http://www.w3.org/2001/XMLSchema#".to_string()),
        );

        context.insert(
            "rdf".to_string(),
            Value::String("http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string()),
        );

        context.insert(
            "rdfs".to_string(),
            Value::String("http://www.w3.org/2000/01/rdf-schema#".to_string()),
        );

        Value::Object(context)
    }
}

impl Default for JsonLdSerializer {
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
    fn test_jsonld_serialization_basic() {
        let aspect = create_test_aspect();
        let serializer = JsonLdSerializer::new();
        let jsonld = serializer
            .serialize_to_string(&aspect)
            .expect("serialization should succeed");

        // Verify it's valid JSON
        let parsed: Value = serde_json::from_str(&jsonld).expect("valid JSON");

        // Check @context
        assert!(parsed.get("@context").is_some());

        // Check @id
        assert_eq!(
            parsed.get("@id").and_then(|v| v.as_str()),
            Some("urn:samm:org.example:1.0.0#TestAspect")
        );

        // Check @type
        assert_eq!(
            parsed.get("@type").and_then(|v| v.as_str()),
            Some("samm:Aspect")
        );
    }

    #[test]
    fn test_jsonld_contains_metadata() {
        let aspect = create_test_aspect();
        let serializer = JsonLdSerializer::new();
        let jsonld = serializer
            .serialize_to_string(&aspect)
            .expect("serialization should succeed");

        assert!(jsonld.contains("samm:preferredName"));
        assert!(jsonld.contains("Test Aspect"));
        assert!(jsonld.contains("samm:description"));
        assert!(jsonld.contains("A test aspect"));
        assert!(jsonld.contains("samm:see"));
        assert!(jsonld.contains("https://example.org/docs"));
    }

    #[test]
    fn test_jsonld_contains_properties() {
        let aspect = create_test_aspect();
        let serializer = JsonLdSerializer::new();
        let jsonld = serializer
            .serialize_to_string(&aspect)
            .expect("serialization should succeed");

        assert!(jsonld.contains("samm:properties"));
        assert!(jsonld.contains("urn:samm:org.example:1.0.0#property1"));
    }

    #[test]
    fn test_jsonld_context_namespaces() {
        let aspect = create_test_aspect();
        let serializer = JsonLdSerializer::new();
        let jsonld = serializer
            .serialize_to_string(&aspect)
            .expect("serialization should succeed");

        let parsed: Value = serde_json::from_str(&jsonld).expect("valid JSON");
        let context = parsed.get("@context").expect("key should exist");

        assert!(context.get("samm").is_some());
        assert!(context.get("samm-c").is_some());
        assert!(context.get("samm-e").is_some());
        assert!(context.get("xsd").is_some());
        assert!(context.get("rdf").is_some());
        assert!(context.get("rdfs").is_some());
    }

    #[test]
    fn test_jsonld_pretty_vs_compact() {
        let aspect = create_test_aspect();

        let pretty = JsonLdSerializer::new().with_pretty(true);
        let compact = JsonLdSerializer::new().with_pretty(false);

        let pretty_output = pretty
            .serialize_to_string(&aspect)
            .expect("serialization should succeed");
        let compact_output = compact
            .serialize_to_string(&aspect)
            .expect("serialization should succeed");

        // Pretty output should have newlines and indentation
        assert!(pretty_output.contains('\n'));
        assert!(pretty_output.len() > compact_output.len());

        // Compact should not have newlines (except possibly in strings)
        assert_eq!(compact_output.matches('\n').count(), 0);
    }

    #[test]
    fn test_jsonld_multi_language() {
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

        let serializer = JsonLdSerializer::new();
        let jsonld = serializer
            .serialize_to_string(&aspect)
            .expect("serialization should succeed");

        // All languages should be present
        assert!(jsonld.contains("@en"));
        assert!(jsonld.contains("@de"));
        assert!(jsonld.contains("@fr"));
        assert!(jsonld.contains("Test Aspect"));
        assert!(jsonld.contains("Test Aspekt"));
        assert!(jsonld.contains("Aspect de Test"));
        assert!(jsonld.contains("English description"));
        assert!(jsonld.contains("Deutsche Beschreibung"));
    }

    #[test]
    fn test_jsonld_empty_aspect() {
        let metadata = ElementMetadata::new("urn:samm:org.example:1.0.0#Empty".to_string());
        let aspect = Aspect {
            metadata,
            properties: vec![],
            operations: vec![],
            events: vec![],
        };

        let serializer = JsonLdSerializer::new();
        let jsonld = serializer
            .serialize_to_string(&aspect)
            .expect("serialization should succeed");

        let parsed: Value = serde_json::from_str(&jsonld).expect("valid JSON");

        assert!(parsed.get("@context").is_some());
        assert!(parsed.get("@id").is_some());
        assert!(parsed.get("@type").is_some());

        // Empty aspect should not have properties/operations/events
        assert!(parsed.get("samm:properties").is_none());
        assert!(parsed.get("samm:operations").is_none());
        assert!(parsed.get("samm:events").is_none());
    }

    #[test]
    fn test_jsonld_with_optional_property() {
        let metadata = ElementMetadata::new("urn:samm:org.example:1.0.0#TestAspect".to_string());

        let mut prop1 = Property::new("urn:samm:org.example:1.0.0#optional1".to_string());
        prop1.optional = true;

        let mut prop2 = Property::new("urn:samm:org.example:1.0.0#required1".to_string());
        prop2.optional = false;

        let aspect = Aspect {
            metadata,
            properties: vec![prop1, prop2],
            operations: vec![],
            events: vec![],
        };

        let serializer = JsonLdSerializer::new();
        let jsonld = serializer
            .serialize_to_string(&aspect)
            .expect("serialization should succeed");

        let parsed: Value = serde_json::from_str(&jsonld).expect("valid JSON");
        let properties = parsed
            .get("samm:properties")
            .expect("key should exist")
            .as_array()
            .expect("should be a valid array");

        // First property should have optional: true
        assert!(properties[0].get("samm:optional").is_some());
        assert_eq!(
            properties[0]
                .get("samm:optional")
                .expect("key should exist")
                .as_bool(),
            Some(true)
        );

        // Second property should not have optional field (default is false)
        assert!(properties[1].get("samm:optional").is_none());
    }

    #[test]
    fn test_jsonld_roundtrip_parse() {
        let aspect = create_test_aspect();
        let serializer = JsonLdSerializer::new();
        let jsonld = serializer
            .serialize_to_string(&aspect)
            .expect("serialization should succeed");

        // Parse it back as JSON to ensure it's valid
        let parsed: Value = serde_json::from_str(&jsonld).expect("valid JSON");

        // Verify key fields
        assert_eq!(
            parsed["@id"].as_str(),
            Some("urn:samm:org.example:1.0.0#TestAspect")
        );
        assert_eq!(parsed["@type"].as_str(), Some("samm:Aspect"));

        // Verify metadata
        assert!(parsed["samm:preferredName"]["@en"].is_string());
        assert!(parsed["samm:description"]["@en"].is_string());
    }
}
