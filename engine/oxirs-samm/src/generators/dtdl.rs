//! SAMM to DTDL Generator
//!
//! Generates DTDL (Digital Twins Definition Language) v3 from SAMM Aspect models.
//! DTDL is the modeling language for Azure Digital Twins.
//!
//! # DTDL v3 Specification
//!
//! - **Context**: `dtmi:dtdl:context;3`
//! - **Interface**: Root element representing a digital twin model
//! - **Contents**: Properties, Telemetry, Commands, Relationships, Components
//!
//! # SAMM to DTDL Mapping
//!
//! | SAMM Element | DTDL Element | Notes |
//! |--------------|--------------|-------|
//! | Aspect | Interface | Root digital twin model |
//! | Property | Property/Telemetry | Based on mutability |
//! | Operation | Command | Function callable on the twin |
//! | Event | Telemetry | Event data emitted by twin |
//! | Entity | Object/Map | Complex nested structure |
//!
//! # URN to DTMI Conversion
//!
//! SAMM URNs are converted to DTMI (Digital Twin Model Identifier) format:
//!
//! ```text
//! urn:samm:com.example:1.0.0#Movement
//! → dtmi:com:example:Movement;1
//! ```
//!
//! # Example Usage
//!
//! ```rust
//! use oxirs_samm::generators::dtdl::generate_dtdl;
//! use oxirs_samm::metamodel::Aspect;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let aspect = Aspect::new("urn:samm:com.example:1.0.0#Movement".to_string());
//! let dtdl = generate_dtdl(&aspect)?;
//! println!("{}", dtdl);
//! # Ok(())
//! # }
//! ```

use crate::error::SammError;
use crate::metamodel::{Aspect, CharacteristicKind, ModelElement, Property};

/// DTDL generation options
#[derive(Debug, Clone)]
pub struct DtdlOptions {
    /// DTDL version (default: 3)
    pub version: u8,
    /// Whether to include descriptions (default: true)
    pub include_descriptions: bool,
    /// Whether to include display names (default: true)
    pub include_display_names: bool,
    /// Whether to generate compact JSON (default: false)
    pub compact: bool,
    /// Whether to mark all properties as writable (default: false)
    pub all_writable: bool,
}

impl Default for DtdlOptions {
    fn default() -> Self {
        Self {
            version: 3,
            include_descriptions: true,
            include_display_names: true,
            compact: false,
            all_writable: false,
        }
    }
}

/// Generate DTDL from SAMM Aspect
///
/// Converts a SAMM Aspect model to a DTDL v3 Interface definition
/// compatible with Azure Digital Twins.
///
/// # Arguments
///
/// * `aspect` - The SAMM Aspect model to convert
///
/// # Returns
///
/// JSON string containing the DTDL Interface definition
///
/// # Example
///
/// ```rust
/// use oxirs_samm::generators::dtdl::generate_dtdl;
/// use oxirs_samm::metamodel::Aspect;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let aspect = Aspect::new("urn:samm:com.example:1.0.0#Movement".to_string());
/// let dtdl = generate_dtdl(&aspect)?;
/// # Ok(())
/// # }
/// ```
pub fn generate_dtdl(aspect: &Aspect) -> Result<String, SammError> {
    generate_dtdl_with_options(aspect, DtdlOptions::default())
}

/// Generate DTDL with custom options
///
/// Provides fine-grained control over DTDL generation.
///
/// # Arguments
///
/// * `aspect` - The SAMM Aspect model to convert
/// * `options` - Generation options
///
/// # Returns
///
/// JSON string containing the DTDL Interface definition
pub fn generate_dtdl_with_options(
    aspect: &Aspect,
    options: DtdlOptions,
) -> Result<String, SammError> {
    let aspect_name = aspect.name();
    let dtmi = to_dtmi(&aspect.metadata.urn)?;

    let indent = if options.compact { "" } else { "  " };
    let nl = if options.compact { "" } else { "\n" };

    let mut dtdl = String::new();

    // Root object
    dtdl.push_str(&format!("{{{nl}"));

    // DTDL context (version 3)
    dtdl.push_str(&format!(
        "{indent}\"@context\": \"dtmi:dtdl:context;{}\",{nl}",
        options.version
    ));

    // Interface ID (DTMI)
    dtdl.push_str(&format!("{indent}\"@id\": \"{}\",{nl}", dtmi));

    // Interface type
    dtdl.push_str(&format!("{indent}\"@type\": \"Interface\",{nl}"));

    // Display name
    if options.include_display_names {
        let display_name = aspect
            .metadata
            .get_preferred_name("en")
            .unwrap_or(&aspect_name);
        dtdl.push_str(&format!(
            "{indent}\"displayName\": \"{}\",{nl}",
            display_name
        ));
    }

    // Description
    if options.include_descriptions {
        if let Some(desc) = aspect.metadata.get_description("en") {
            dtdl.push_str(&format!(
                "{indent}\"description\": \"{}\",{nl}",
                escape_json(desc)
            ));
        }
    }

    // Comment (optional metadata)
    if !aspect.metadata.see_refs.is_empty() {
        let see_also = aspect.metadata.see_refs.join(", ");
        dtdl.push_str(&format!(
            "{indent}\"comment\": \"See also: {}\",{nl}",
            escape_json(&see_also)
        ));
    }

    // Contents array
    dtdl.push_str(&format!("{indent}\"contents\": [{nl}"));

    let mut contents = Vec::new();

    // Convert Properties to DTDL Property/Telemetry
    for prop in aspect.properties() {
        contents.push(generate_property_content(prop, &options, indent, nl)?);
    }

    // Convert Operations to DTDL Commands
    for op in aspect.operations() {
        let op_name = to_camel_case(&op.name());
        let op_default_name = op.name();
        let op_display_name = op
            .metadata
            .get_preferred_name("en")
            .unwrap_or(&op_default_name);

        let mut cmd = format!("{indent}{indent}{{{nl}");
        cmd.push_str(&format!(
            "{indent}{indent}{indent}\"@type\": \"Command\",{nl}"
        ));
        cmd.push_str(&format!(
            "{indent}{indent}{indent}\"name\": \"{}\"",
            op_name
        ));

        if options.include_display_names {
            cmd.push_str(&format!(",{nl}"));
            cmd.push_str(&format!(
                "{indent}{indent}{indent}\"displayName\": \"{}\"",
                op_display_name
            ));
        }

        if options.include_descriptions {
            if let Some(desc) = op.metadata.get_description("en") {
                cmd.push_str(&format!(",{nl}"));
                cmd.push_str(&format!(
                    "{indent}{indent}{indent}\"description\": \"{}\"",
                    escape_json(desc)
                ));
            }
        }

        cmd.push_str(nl);
        cmd.push_str(&format!("{indent}{indent}}}"));
        contents.push(cmd);
    }

    // Convert Events to DTDL Telemetry (with event semantics)
    for event in aspect.events() {
        let event_name = to_camel_case(&event.name());
        let event_default_name = event.name();
        let event_display_name = event
            .metadata
            .get_preferred_name("en")
            .unwrap_or(&event_default_name);

        let mut telemetry = format!("{indent}{indent}{{{nl}");
        telemetry.push_str(&format!(
            "{indent}{indent}{indent}\"@type\": \"Telemetry\",{nl}"
        ));
        telemetry.push_str(&format!(
            "{indent}{indent}{indent}\"name\": \"{}\"",
            event_name
        ));

        if options.include_display_names {
            telemetry.push_str(&format!(",{nl}"));
            telemetry.push_str(&format!(
                "{indent}{indent}{indent}\"displayName\": \"{}\"",
                event_display_name
            ));
        }

        if options.include_descriptions {
            if let Some(desc) = event.metadata.get_description("en") {
                telemetry.push_str(&format!(",{nl}"));
                telemetry.push_str(&format!(
                    "{indent}{indent}{indent}\"description\": \"{}\"",
                    escape_json(desc)
                ));
            }
        }

        // Default to Object schema for events
        telemetry.push_str(&format!(",{nl}"));
        telemetry.push_str(&format!(
            "{indent}{indent}{indent}\"schema\": \"dtmi:dtdl:instance:Schema:Object;3\"{nl}"
        ));

        telemetry.push_str(&format!("{indent}{indent}}}"));
        contents.push(telemetry);
    }

    // Join contents with commas
    dtdl.push_str(&contents.join(&format!(",{nl}")));
    dtdl.push_str(nl);

    dtdl.push_str(&format!("{indent}]{nl}"));
    dtdl.push_str(&format!("}}{nl}"));

    Ok(dtdl)
}

/// Generate DTDL content for a SAMM Property
fn generate_property_content(
    prop: &Property,
    options: &DtdlOptions,
    indent: &str,
    nl: &str,
) -> Result<String, SammError> {
    let prop_name = to_camel_case(&prop.effective_name());
    let default_name = prop.name();
    let prop_display_name = prop
        .metadata
        .get_preferred_name("en")
        .unwrap_or(&default_name);

    let mut content = format!("{indent}{indent}{{{nl}");

    // Determine if this is a Property or Telemetry
    // Properties are writable state, Telemetry is read-only data
    let is_writable = options.all_writable || !prop.optional;
    let content_type = if is_writable { "Property" } else { "Telemetry" };

    content.push_str(&format!(
        "{indent}{indent}{indent}\"@type\": \"{}\",{nl}",
        content_type
    ));
    content.push_str(&format!(
        "{indent}{indent}{indent}\"name\": \"{}\"",
        prop_name
    ));

    if options.include_display_names {
        content.push_str(&format!(",{nl}"));
        content.push_str(&format!(
            "{indent}{indent}{indent}\"displayName\": \"{}\"",
            prop_display_name
        ));
    }

    if options.include_descriptions {
        if let Some(desc) = prop.metadata.get_description("en") {
            content.push_str(&format!(",{nl}"));
            content.push_str(&format!(
                "{indent}{indent}{indent}\"description\": \"{}\"",
                escape_json(desc)
            ));
        }
    }

    // Schema (data type)
    let schema = if let Some(characteristic) = &prop.characteristic {
        map_characteristic_to_schema(characteristic)?
    } else {
        "string".to_string()
    };

    content.push_str(&format!(",{nl}"));
    content.push_str(&format!(
        "{indent}{indent}{indent}\"schema\": \"{}\"{nl}",
        schema
    ));

    content.push_str(&format!("{indent}{indent}}}"));

    Ok(content)
}

/// Map SAMM Characteristic to DTDL schema type
fn map_characteristic_to_schema(
    characteristic: &crate::metamodel::Characteristic,
) -> Result<String, SammError> {
    // Check data_type first
    if let Some(data_type) = &characteristic.data_type {
        return map_xsd_to_dtdl_schema(data_type);
    }

    // Map based on CharacteristicKind
    match &characteristic.kind {
        CharacteristicKind::Trait => Ok("string".to_string()),
        CharacteristicKind::Quantifiable { .. } => Ok("double".to_string()),
        CharacteristicKind::Measurement { .. } => Ok("double".to_string()),
        CharacteristicKind::Enumeration { values } => {
            // DTDL doesn't have native enums, use string
            // In production, could generate an Enum schema separately
            Ok("string".to_string())
        }
        CharacteristicKind::State { .. } => Ok("string".to_string()),
        CharacteristicKind::Duration { .. } => Ok("duration".to_string()),
        CharacteristicKind::Collection { .. } => {
            // DTDL Array type
            Ok("dtmi:dtdl:instance:Schema:Array;3".to_string())
        }
        CharacteristicKind::List { .. } => Ok("dtmi:dtdl:instance:Schema:Array;3".to_string()),
        CharacteristicKind::Set { .. } => Ok("dtmi:dtdl:instance:Schema:Array;3".to_string()),
        CharacteristicKind::SortedSet { .. } => Ok("dtmi:dtdl:instance:Schema:Array;3".to_string()),
        CharacteristicKind::TimeSeries { .. } => {
            Ok("dtmi:dtdl:instance:Schema:Array;3".to_string())
        }
        CharacteristicKind::Code => Ok("string".to_string()),
        CharacteristicKind::Either { .. } => {
            // DTDL doesn't have union types, use Object
            Ok("dtmi:dtdl:instance:Schema:Object;3".to_string())
        }
        CharacteristicKind::SingleEntity { .. } => {
            Ok("dtmi:dtdl:instance:Schema:Object;3".to_string())
        }
        CharacteristicKind::StructuredValue { .. } => {
            Ok("dtmi:dtdl:instance:Schema:Object;3".to_string())
        }
    }
}

/// Map XSD data type to DTDL schema
fn map_xsd_to_dtdl_schema(xsd_type: &str) -> Result<String, SammError> {
    // Extract the base type name from various XSD formats
    // Examples: "xsd:int" → "int", "http://...#double" → "double", "int" → "int"
    let after_hash = xsd_type.split('#').next_back().unwrap_or(xsd_type);
    let base_type = after_hash.split(':').next_back().unwrap_or(after_hash);

    let dtdl_type = match base_type {
        // Numeric types
        "int" | "integer" | "short" | "byte" => "integer",
        "long" => "long",
        "float" | "decimal" => "float",
        "double" => "double",

        // Boolean
        "boolean" | "bool" => "boolean",

        // String types
        "string" | "normalizedString" | "token" | "language" | "Name" | "NCName" => "string",

        // Date/Time types
        "dateTime" => "dateTime",
        "date" => "date",
        "time" => "time",
        "duration" => "duration",

        // Default to string
        _ => "string",
    };

    Ok(dtdl_type.to_string())
}

/// Convert SAMM URN to DTMI (Digital Twin Model Identifier)
///
/// Converts a SAMM URN to DTMI format for DTDL.
///
/// # Format
///
/// ```text
/// urn:samm:com.example:1.0.0#Movement
/// → dtmi:com:example:Movement;1
/// ```
///
/// # Arguments
///
/// * `urn` - SAMM URN string
///
/// # Returns
///
/// DTMI string
fn to_dtmi(urn: &str) -> Result<String, SammError> {
    // urn:samm:com.example:1.0.0#Movement
    // → dtmi:com:example:Movement;1

    if !urn.starts_with("urn:samm:") {
        return Err(SammError::Generation(format!(
            "Invalid SAMM URN: {}. Expected format: urn:samm:namespace:version#name",
            urn
        )));
    }

    let without_prefix = urn.strip_prefix("urn:samm:").unwrap();

    // Split by '#'
    let parts: Vec<&str> = without_prefix.split('#').collect();
    if parts.len() != 2 {
        return Err(SammError::Generation(format!(
            "Invalid SAMM URN format: {}. Expected '#' separator",
            urn
        )));
    }

    let namespace_version = parts[0];
    let name = parts[1];

    // Split namespace and version
    // com.example:1.0.0 → com.example and 1.0.0
    let ns_parts: Vec<&str> = namespace_version.rsplitn(2, ':').collect();
    if ns_parts.len() != 2 {
        return Err(SammError::Generation(format!(
            "Invalid SAMM URN format: {}. Expected ':' separator for version",
            urn
        )));
    }

    let version = ns_parts[0];
    let namespace = ns_parts[1];

    // Convert dots to colons in namespace (com.example → com:example)
    let namespace_dtmi = namespace.replace('.', ":");

    // Extract major version (1.0.0 → 1)
    let major_version = version
        .split('.')
        .next()
        .ok_or_else(|| SammError::Generation(format!("Invalid version: {}", version)))?;

    // Build DTMI
    let dtmi = format!("dtmi:{}:{};{}", namespace_dtmi, name, major_version);

    Ok(dtmi)
}

/// Convert snake_case or kebab-case to camelCase
fn to_camel_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;

    for ch in s.chars() {
        if ch == '_' || ch == '-' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(ch.to_uppercase().next().unwrap());
            capitalize_next = false;
        } else {
            result.push(ch);
        }
    }

    result
}

/// Escape JSON string
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Characteristic, ElementMetadata, Operation};

    #[test]
    fn test_to_dtmi_conversion() {
        assert_eq!(
            to_dtmi("urn:samm:com.example:1.0.0#Movement").unwrap(),
            "dtmi:com:example:Movement;1"
        );

        assert_eq!(
            to_dtmi("urn:samm:org.eclipse.esmf:2.3.0#Aspect").unwrap(),
            "dtmi:org:eclipse:esmf:Aspect;2"
        );

        assert_eq!(
            to_dtmi("urn:samm:io.github.oxirs:0.1.0#TestAspect").unwrap(),
            "dtmi:io:github:oxirs:TestAspect;0"
        );
    }

    #[test]
    fn test_to_dtmi_invalid_urn() {
        assert!(to_dtmi("invalid:urn").is_err());
        assert!(to_dtmi("urn:samm:no-hash").is_err());
        assert!(to_dtmi("urn:samm:no-version#Name").is_err());
    }

    #[test]
    fn test_camel_case_conversion() {
        assert_eq!(to_camel_case("movement_aspect"), "movementAspect");
        assert_eq!(to_camel_case("position"), "position");
        assert_eq!(to_camel_case("current_speed"), "currentSpeed");
        assert_eq!(to_camel_case("current-speed"), "currentSpeed");
        assert_eq!(to_camel_case("GPS_coordinates"), "GPSCoordinates");
    }

    #[test]
    fn test_xsd_to_dtdl_schema_mapping() {
        assert_eq!(map_xsd_to_dtdl_schema("string").unwrap(), "string");
        assert_eq!(map_xsd_to_dtdl_schema("int").unwrap(), "integer");
        assert_eq!(map_xsd_to_dtdl_schema("xsd:int").unwrap(), "integer");
        assert_eq!(map_xsd_to_dtdl_schema("float").unwrap(), "float");
        assert_eq!(map_xsd_to_dtdl_schema("xsd:float").unwrap(), "float");
        assert_eq!(map_xsd_to_dtdl_schema("boolean").unwrap(), "boolean");
        assert_eq!(map_xsd_to_dtdl_schema("dateTime").unwrap(), "dateTime");
        assert_eq!(
            map_xsd_to_dtdl_schema("http://www.w3.org/2001/XMLSchema#double").unwrap(),
            "double"
        );
    }

    #[test]
    fn test_escape_json() {
        assert_eq!(escape_json("hello"), "hello");
        assert_eq!(escape_json("hello \"world\""), "hello \\\"world\\\"");
        assert_eq!(escape_json("line1\nline2"), "line1\\nline2");
        assert_eq!(escape_json("tab\there"), "tab\\there");
    }

    #[test]
    fn test_basic_aspect_generation() {
        let aspect = Aspect::new("urn:samm:com.example:1.0.0#Movement".to_string());
        let dtdl = generate_dtdl(&aspect).unwrap();

        assert!(dtdl.contains("\"@context\": \"dtmi:dtdl:context;3\""));
        assert!(dtdl.contains("\"@id\": \"dtmi:com:example:Movement;1\""));
        assert!(dtdl.contains("\"@type\": \"Interface\""));
    }

    #[test]
    fn test_aspect_with_property() {
        let mut aspect = Aspect::new("urn:samm:com.example:1.0.0#Movement".to_string());

        let mut prop = Property::new("urn:samm:com.example:1.0.0#speed".to_string());
        let mut char = Characteristic::new(
            "urn:samm:com.example:1.0.0#SpeedCharacteristic".to_string(),
            CharacteristicKind::Measurement {
                unit: "unit:kilometrePerHour".to_string(),
            },
        );
        char.data_type = Some("xsd:float".to_string());
        prop.characteristic = Some(char);

        aspect.add_property(prop);

        let dtdl = generate_dtdl(&aspect).unwrap();

        assert!(dtdl.contains("\"name\": \"speed\""));
        assert!(dtdl.contains("\"schema\": \"float\""));
    }

    #[test]
    fn test_aspect_with_operation() {
        let mut aspect = Aspect::new("urn:samm:com.example:1.0.0#Movement".to_string());

        let op = Operation::new("urn:samm:com.example:1.0.0#stop".to_string());
        aspect.add_operation(op);

        let dtdl = generate_dtdl(&aspect).unwrap();

        assert!(dtdl.contains("\"@type\": \"Command\""));
        assert!(dtdl.contains("\"name\": \"stop\""));
    }

    #[test]
    fn test_compact_output() {
        let aspect = Aspect::new("urn:samm:com.example:1.0.0#Movement".to_string());
        let options = DtdlOptions {
            compact: true,
            ..Default::default()
        };
        let dtdl = generate_dtdl_with_options(&aspect, options).unwrap();

        // Compact JSON should not have indentation
        assert!(!dtdl.contains("  "));
    }

    #[test]
    fn test_options_include_descriptions() {
        let mut aspect = Aspect::new("urn:samm:com.example:1.0.0#Movement".to_string());
        aspect
            .metadata
            .add_description("en".to_string(), "Movement tracking aspect".to_string());

        let options = DtdlOptions {
            include_descriptions: true,
            ..Default::default()
        };
        let dtdl = generate_dtdl_with_options(&aspect, options).unwrap();

        assert!(dtdl.contains("\"description\": \"Movement tracking aspect\""));

        let options_no_desc = DtdlOptions {
            include_descriptions: false,
            ..Default::default()
        };
        let dtdl_no_desc = generate_dtdl_with_options(&aspect, options_no_desc).unwrap();

        assert!(!dtdl_no_desc.contains("\"description\""));
    }
}
