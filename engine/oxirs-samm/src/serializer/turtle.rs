//! Turtle (TTL) serializer for SAMM models
//!
//! Serializes SAMM Aspect models to Turtle/RDF format following SAMM 2.1.0+ specification.

use crate::error::{Result, SammError};
use crate::metamodel::{
    Aspect, Characteristic, CharacteristicKind, ModelElement, Operation, Property,
};
use std::path::Path;
use tokio::fs;

/// Turtle serializer for SAMM models
pub struct TurtleSerializer {
    /// Indentation level
    indent_size: usize,
}

impl TurtleSerializer {
    /// Create a new Turtle serializer
    pub fn new() -> Self {
        Self { indent_size: 2 }
    }

    /// Serialize an Aspect to a Turtle file
    pub async fn serialize_to_file(&self, aspect: &Aspect, path: &Path) -> Result<()> {
        let content = self.serialize_to_string(aspect)?;

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        fs::write(path, content).await?;

        Ok(())
    }

    /// Serialize an Aspect to a Turtle string
    pub fn serialize_to_string(&self, aspect: &Aspect) -> Result<String> {
        let mut output = String::new();

        // Add prefixes
        output.push_str(&self.generate_prefixes());
        output.push('\n');

        // Add Aspect definition
        output.push_str(&self.serialize_aspect(aspect)?);

        // Add Properties
        for property in aspect.properties() {
            output.push('\n');
            output.push_str(&self.serialize_property(property)?);
        }

        // Add Operations
        for operation in aspect.operations() {
            output.push('\n');
            output.push_str(&self.serialize_operation(operation)?);
        }

        Ok(output)
    }

    /// Generate RDF prefixes
    fn generate_prefixes(&self) -> String {
        let mut prefixes = String::new();

        prefixes.push_str("@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#> .\n");
        prefixes
            .push_str("@prefix samm-c: <urn:samm:org.eclipse.esmf.samm:characteristic:2.1.0#> .\n");
        prefixes.push_str("@prefix samm-e: <urn:samm:org.eclipse.esmf.samm:entity:2.1.0#> .\n");
        prefixes.push_str("@prefix unit: <urn:samm:org.eclipse.esmf.samm:unit:2.1.0#> .\n");
        prefixes.push_str("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n");
        prefixes.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n");
        prefixes.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n");

        prefixes
    }

    /// Serialize an Aspect
    fn serialize_aspect(&self, aspect: &Aspect) -> Result<String> {
        let mut output = String::new();
        let metadata = aspect.metadata();

        output.push_str(&format!("<{}> a samm:Aspect", metadata.urn));

        // Add preferred names
        if !metadata.preferred_names.is_empty() {
            for (lang, name) in &metadata.preferred_names {
                output.push_str(&format!(
                    " ;\n  samm:preferredName \"{}\"@{}",
                    self.escape_string(name),
                    lang
                ));
            }
        }

        // Add descriptions
        if !metadata.descriptions.is_empty() {
            for (lang, desc) in &metadata.descriptions {
                output.push_str(&format!(
                    " ;\n  samm:description \"{}\"@{}",
                    self.escape_string(desc),
                    lang
                ));
            }
        }

        // Add properties
        if !aspect.properties().is_empty() {
            output.push_str(" ;\n  samm:properties (");
            for (i, prop) in aspect.properties().iter().enumerate() {
                if i > 0 {
                    output.push(' ');
                }
                output.push_str(&format!("<{}>", prop.metadata().urn));
            }
            output.push(')');
        }

        // Add operations
        if !aspect.operations().is_empty() {
            output.push_str(" ;\n  samm:operations (");
            for (i, op) in aspect.operations().iter().enumerate() {
                if i > 0 {
                    output.push(' ');
                }
                output.push_str(&format!("<{}>", op.metadata().urn));
            }
            output.push(')');
        }

        output.push_str(" .\n");
        Ok(output)
    }

    /// Serialize a Property
    fn serialize_property(&self, property: &Property) -> Result<String> {
        let mut output = String::new();
        let metadata = property.metadata();

        output.push_str(&format!("<{}> a samm:Property", metadata.urn));

        // Add preferred names
        if !metadata.preferred_names.is_empty() {
            for (lang, name) in &metadata.preferred_names {
                output.push_str(&format!(
                    " ;\n  samm:preferredName \"{}\"@{}",
                    self.escape_string(name),
                    lang
                ));
            }
        }

        // Add descriptions
        if !metadata.descriptions.is_empty() {
            for (lang, desc) in &metadata.descriptions {
                output.push_str(&format!(
                    " ;\n  samm:description \"{}\"@{}",
                    self.escape_string(desc),
                    lang
                ));
            }
        }

        // Add characteristic
        if let Some(characteristic) = &property.characteristic {
            output.push_str(&format!(
                " ;\n  samm:characteristic <{}>",
                characteristic.urn()
            ));
        }

        // Add optional flag
        if property.optional {
            output.push_str(" ;\n  samm:optional \"true\"^^xsd:boolean");
        }

        output.push_str(" .\n");

        // Serialize characteristic if present
        if let Some(characteristic) = &property.characteristic {
            output.push('\n');
            output.push_str(&self.serialize_characteristic(characteristic)?);
        }

        Ok(output)
    }

    /// Serialize an Operation
    fn serialize_operation(&self, operation: &Operation) -> Result<String> {
        let mut output = String::new();
        let metadata = operation.metadata();

        output.push_str(&format!("<{}> a samm:Operation", metadata.urn));

        // Add preferred names
        if !metadata.preferred_names.is_empty() {
            for (lang, name) in &metadata.preferred_names {
                output.push_str(&format!(
                    " ;\n  samm:preferredName \"{}\"@{}",
                    self.escape_string(name),
                    lang
                ));
            }
        }

        // Add descriptions
        if !metadata.descriptions.is_empty() {
            for (lang, desc) in &metadata.descriptions {
                output.push_str(&format!(
                    " ;\n  samm:description \"{}\"@{}",
                    self.escape_string(desc),
                    lang
                ));
            }
        }

        // Add input parameters
        if !operation.input.is_empty() {
            output.push_str(" ;\n  samm:input (");
            for (i, prop) in operation.input.iter().enumerate() {
                if i > 0 {
                    output.push(' ');
                }
                output.push_str(&format!("<{}>", prop.metadata().urn));
            }
            output.push(')');
        }

        // Add output
        if let Some(output_prop) = &operation.output {
            output.push_str(&format!(
                " ;\n  samm:output <{}>",
                output_prop.metadata().urn
            ));
        }

        output.push_str(" .\n");
        Ok(output)
    }

    /// Serialize a Characteristic
    fn serialize_characteristic(&self, characteristic: &Characteristic) -> Result<String> {
        let mut output = String::new();

        let char_type = match &characteristic.kind {
            CharacteristicKind::Trait => "samm-c:Trait",
            CharacteristicKind::Quantifiable { .. } => "samm-c:Quantifiable",
            CharacteristicKind::Measurement { .. } => "samm-c:Measurement",
            CharacteristicKind::Enumeration { .. } => "samm-c:Enumeration",
            CharacteristicKind::State { .. } => "samm-c:State",
            CharacteristicKind::Duration { .. } => "samm-c:Duration",
            CharacteristicKind::Collection { .. } => "samm-c:Collection",
            CharacteristicKind::List { .. } => "samm-c:List",
            CharacteristicKind::Set { .. } => "samm-c:Set",
            CharacteristicKind::SortedSet { .. } => "samm-c:SortedSet",
            CharacteristicKind::TimeSeries { .. } => "samm-c:TimeSeries",
            CharacteristicKind::Either { .. } => "samm-c:Either",
            CharacteristicKind::SingleEntity { .. } => "samm-c:SingleEntity",
            CharacteristicKind::StructuredValue { .. } => "samm-c:StructuredValue",
            CharacteristicKind::Code => "samm-c:Code",
        };

        output.push_str(&format!("<{}> a {}", characteristic.urn(), char_type));

        // Add data type if present
        if let Some(data_type) = &characteristic.data_type {
            output.push_str(&format!(" ;\n  samm:dataType <{}>", data_type));
        }

        output.push_str(" .\n");
        Ok(output)
    }

    /// Escape special characters in strings
    fn escape_string(&self, s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    }
}

impl Default for TurtleSerializer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{CharacteristicKind, ElementMetadata};

    #[test]
    fn test_serialize_minimal_aspect() {
        let mut metadata = ElementMetadata::new("urn:test:aspect#MinimalAspect".to_string());
        metadata.add_preferred_name("en".to_string(), "Minimal Aspect".to_string());

        let aspect = Aspect {
            metadata,
            properties: vec![],
            operations: vec![],
            events: vec![],
        };

        let serializer = TurtleSerializer::new();
        let result = serializer.serialize_to_string(&aspect);

        assert!(result.is_ok());
        let ttl = result.unwrap();

        assert!(ttl.contains("@prefix samm:"));
        assert!(ttl.contains("<urn:test:aspect#MinimalAspect> a samm:Aspect"));
        assert!(ttl.contains("samm:preferredName \"Minimal Aspect\"@en"));
    }

    #[test]
    fn test_serialize_aspect_with_property() {
        let mut aspect_meta = ElementMetadata::new("urn:test:aspect#TestAspect".to_string());
        aspect_meta.add_preferred_name("en".to_string(), "Test Aspect".to_string());

        let mut prop_meta = ElementMetadata::new("urn:test:property#temperature".to_string());
        prop_meta.add_preferred_name("en".to_string(), "temperature".to_string());

        let characteristic = Characteristic::new(
            "urn:test:characteristic#TemperatureCharacteristic".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("xsd:float".to_string());

        let property = Property {
            metadata: prop_meta,
            characteristic: Some(characteristic),
            example_values: vec![],
            optional: false,
            is_collection: false,
            payload_name: None,
            is_abstract: false,
            extends: None,
        };

        let mut aspect = Aspect {
            metadata: aspect_meta,
            properties: vec![],
            operations: vec![],
            events: vec![],
        };
        aspect.add_property(property);

        let serializer = TurtleSerializer::new();
        let result = serializer.serialize_to_string(&aspect);

        assert!(result.is_ok());
        let ttl = result.unwrap();

        assert!(ttl.contains("<urn:test:aspect#TestAspect> a samm:Aspect"));
        assert!(ttl.contains("samm:properties"));
        assert!(ttl.contains("<urn:test:property#temperature>"));
        assert!(ttl.contains("a samm:Property"));
        assert!(ttl.contains("samm:characteristic"));
    }

    #[test]
    fn test_escape_string() {
        let serializer = TurtleSerializer::new();

        assert_eq!(serializer.escape_string("test\"quote"), "test\\\"quote");
        assert_eq!(serializer.escape_string("test\\slash"), "test\\\\slash");
        assert_eq!(serializer.escape_string("test\nline"), "test\\nline");
    }
}
