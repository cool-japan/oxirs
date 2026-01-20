//! SAMM to TypeScript Interface Generator
//!
//! Generates TypeScript type definitions and interfaces from SAMM Aspect models.
//! Supports enums, nested types, JSDoc comments, and various naming conventions.

use crate::error::SammError;
use crate::metamodel::{Aspect, CharacteristicKind, ModelElement};
use std::collections::HashSet;

/// TypeScript generation options
#[derive(Debug, Clone)]
pub struct TsOptions {
    /// Export types as default
    pub export_default: bool,
    /// Use strict null checks (T | undefined for optional fields)
    pub strict_null_checks: bool,
    /// Make all properties readonly
    pub readonly_properties: bool,
    /// Convert snake_case to camelCase
    pub snake_case_to_camel: bool,
}

impl Default for TsOptions {
    fn default() -> Self {
        Self {
            export_default: false,
            strict_null_checks: true,
            readonly_properties: false,
            snake_case_to_camel: true,
        }
    }
}

/// Generate TypeScript interfaces from SAMM Aspect
pub fn generate_typescript(aspect: &Aspect, options: TsOptions) -> Result<String, SammError> {
    let mut ts = String::new();
    let mut enum_types = Vec::new();
    let mut nested_types = HashSet::new();

    // Header comment
    ts.push_str("/**\n");
    ts.push_str(&format!(" * TypeScript interfaces for {}\n", aspect.name()));
    ts.push_str(" * Generated from SAMM Aspect Model\n");
    ts.push_str(&format!(" * URN: {}\n", aspect.metadata().urn));
    if let Some(desc) = aspect.metadata().get_description("en") {
        ts.push_str(&format!(" * \n * {}\n", desc));
    }
    ts.push_str(" */\n\n");

    // Collect enum types and nested types
    for prop in aspect.properties() {
        if let Some(char) = &prop.characteristic {
            // Collect enum types
            if let CharacteristicKind::Enumeration { values } = char.kind() {
                let enum_name = to_pascal_case(&prop.name());
                let enum_def = generate_enum_type(&enum_name, values, &options);
                enum_types.push(enum_def);
            }

            // Collect state types
            if let CharacteristicKind::State {
                values,
                default_value: _,
            } = char.kind()
            {
                let enum_name = to_pascal_case(&prop.name());
                let enum_def = generate_enum_type(&enum_name, values, &options);
                enum_types.push(enum_def);
            }

            // Collect entity types
            if let CharacteristicKind::SingleEntity { entity_type } = char.kind() {
                let entity_name = entity_type.split('#').next_back().unwrap_or(entity_type);
                nested_types.insert(to_pascal_case(entity_name));
            }
        }
    }

    // Generate enum types
    for enum_def in &enum_types {
        ts.push_str(enum_def);
        ts.push_str("\n\n");
    }

    // Generate main interface
    let main_interface = generate_main_interface(aspect, &options)?;
    ts.push_str(&main_interface);

    // Generate nested type stubs if any
    for nested_type in nested_types {
        ts.push_str("\n\n");
        ts.push_str(&format!(
            "/**\n * Referenced entity type: {}\n",
            nested_type
        ));
        ts.push_str(" * This is a stub - implement based on your entity structure\n */\n");
        ts.push_str(&format!("export interface {} {{\n", nested_type));
        ts.push_str("  id: string;\n");
        ts.push_str("  // Add other properties as needed\n");
        ts.push_str("}\n");
    }

    Ok(ts)
}

/// Generate enum type
fn generate_enum_type(name: &str, values: &[String], options: &TsOptions) -> String {
    let mut enum_def = String::new();

    enum_def.push_str(&format!("/**\n * Enumeration: {}\n */\n", name));
    enum_def.push_str(&format!("export enum {} {{\n", name));

    for (i, value) in values.iter().enumerate() {
        // Convert to PascalCase for enum member name
        let member_name = to_pascal_case(value);
        let comma = if i < values.len() - 1 { "," } else { "" };
        enum_def.push_str(&format!("  {} = \"{}\"{}\n", member_name, value, comma));
    }

    enum_def.push('}');
    enum_def
}

/// Generate main interface
fn generate_main_interface(aspect: &Aspect, options: &TsOptions) -> Result<String, SammError> {
    let type_name = to_pascal_case(&aspect.name());
    let mut interface_def = String::new();

    // JSDoc comment
    interface_def.push_str("/**\n");
    interface_def.push_str(&format!(" * {} interface\n", type_name));
    if let Some(desc) = aspect.metadata().get_description("en") {
        interface_def.push_str(&format!(" * \n * {}\n", desc));
    }
    interface_def.push_str(" * \n * @generated from SAMM model\n");
    interface_def.push_str(" */\n");

    interface_def.push_str(&format!("export interface {} {{\n", type_name));

    // Always add id field
    let readonly = if options.readonly_properties {
        "readonly "
    } else {
        ""
    };
    interface_def.push_str("  /** Unique identifier */\n");
    interface_def.push_str(&format!("  {}id: string;\n\n", readonly));

    // Add properties
    for prop in aspect.properties() {
        let field_name = if options.snake_case_to_camel {
            to_camel_case(&prop.name())
        } else {
            prop.name().to_string()
        };

        let field_type = get_typescript_type(prop, options)?;

        // JSDoc comment for property
        if let Some(desc) = prop.metadata().get_description("en") {
            interface_def.push_str(&format!("  /**\n   * {}\n   */\n", desc));
        }

        interface_def.push_str(&format!(
            "  {}{}: {};\n\n",
            readonly, field_name, field_type
        ));
    }

    // Add metadata fields
    interface_def.push_str("  /** Timestamp of creation */\n");
    interface_def.push_str(&format!("  {}createdAt: Date | string;\n\n", readonly));
    interface_def.push_str("  /** Timestamp of last update */\n");
    interface_def.push_str(&format!("  {}updatedAt: Date | string;\n", readonly));

    interface_def.push_str("}\n");

    Ok(interface_def)
}

/// Get TypeScript type for a property
fn get_typescript_type(
    prop: &crate::metamodel::Property,
    options: &TsOptions,
) -> Result<String, SammError> {
    if let Some(char) = &prop.characteristic {
        let base_type = match char.kind() {
            CharacteristicKind::Enumeration { .. } | CharacteristicKind::State { .. } => {
                // Use the property name as the enum type name
                to_pascal_case(&prop.name())
            }
            CharacteristicKind::Collection { .. } | CharacteristicKind::List { .. } => {
                // Array type
                if let Some(dt) = &char.data_type {
                    let element_type = map_xsd_to_typescript(dt);
                    format!("Array<{}>", element_type)
                } else {
                    "Array<string>".to_string()
                }
            }
            CharacteristicKind::Set { .. } | CharacteristicKind::SortedSet { .. } => {
                // Set type (represented as array in TypeScript)
                if let Some(dt) = &char.data_type {
                    let element_type = map_xsd_to_typescript(dt);
                    format!("Array<{}>", element_type)
                } else {
                    "Array<string>".to_string()
                }
            }
            CharacteristicKind::SingleEntity { entity_type } => {
                // Object reference - extract entity name
                let entity_name = entity_type.split('#').next_back().unwrap_or(entity_type);
                to_pascal_case(entity_name)
            }
            CharacteristicKind::TimeSeries { .. } => {
                // Time series as array of objects
                "Array<{ timestamp: Date | string; value: number }>".to_string()
            }
            CharacteristicKind::Measurement { unit: _ }
            | CharacteristicKind::Quantifiable { .. } => {
                // Numeric value with unit metadata
                if let Some(dt) = &char.data_type {
                    map_xsd_to_typescript(dt)
                } else {
                    "number".to_string()
                }
            }
            _ => {
                // Default: use data type
                if let Some(dt) = &char.data_type {
                    map_xsd_to_typescript(dt)
                } else {
                    "string".to_string()
                }
            }
        };

        // Add undefined for optional fields with strict null checks
        if prop.optional && options.strict_null_checks {
            Ok(format!("{} | undefined", base_type))
        } else {
            Ok(base_type)
        }
    } else {
        // No characteristic, default to string
        if prop.optional && options.strict_null_checks {
            Ok("string | undefined".to_string())
        } else {
            Ok("string".to_string())
        }
    }
}

/// Map XSD types to TypeScript types
fn map_xsd_to_typescript(xsd_type: &str) -> String {
    match xsd_type {
        t if t.ends_with("string") => "string".to_string(),
        t if t.ends_with("int") | t.ends_with("integer") => "number".to_string(),
        t if t.ends_with("long") => "number".to_string(),
        t if t.ends_with("short") | t.ends_with("byte") => "number".to_string(),
        t if t.ends_with("decimal") => "number".to_string(),
        t if t.ends_with("float") => "number".to_string(),
        t if t.ends_with("double") => "number".to_string(),
        t if t.ends_with("boolean") => "boolean".to_string(),
        t if t.ends_with("date") => "Date | string".to_string(),
        t if t.ends_with("dateTime") | t.ends_with("dateTimeStamp") => "Date | string".to_string(),
        t if t.ends_with("time") => "string".to_string(),
        t if t.ends_with("duration") => "string".to_string(),
        t if t.ends_with("anyURI") => "string".to_string(),
        t if t.ends_with("base64Binary") => "string".to_string(),
        t if t.ends_with("hexBinary") => "string".to_string(),
        _ => "string".to_string(),
    }
}

/// Convert snake_case or PascalCase to camelCase
fn to_camel_case(s: &str) -> String {
    let pascal = to_pascal_case(s);
    if pascal.is_empty() {
        return pascal;
    }

    let mut chars = pascal.chars();
    let first = chars.next().unwrap().to_lowercase().to_string();
    format!("{}{}", first, chars.as_str())
}

/// Convert snake_case or camelCase to PascalCase
fn to_pascal_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;

    for ch in s.chars() {
        if ch == '_' || ch == '-' || ch == ' ' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(ch.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(ch);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xsd_to_typescript_mapping() {
        assert_eq!(
            map_xsd_to_typescript("http://www.w3.org/2001/XMLSchema#string"),
            "string"
        );
        assert_eq!(
            map_xsd_to_typescript("http://www.w3.org/2001/XMLSchema#int"),
            "number"
        );
        assert_eq!(
            map_xsd_to_typescript("http://www.w3.org/2001/XMLSchema#boolean"),
            "boolean"
        );
        assert_eq!(
            map_xsd_to_typescript("http://www.w3.org/2001/XMLSchema#float"),
            "number"
        );
        assert_eq!(
            map_xsd_to_typescript("http://www.w3.org/2001/XMLSchema#dateTime"),
            "Date | string"
        );
    }

    #[test]
    fn test_case_conversion() {
        assert_eq!(to_camel_case("MovementAspect"), "movementAspect");
        assert_eq!(to_camel_case("current_speed"), "currentSpeed");
        assert_eq!(to_pascal_case("movement_aspect"), "MovementAspect");
        assert_eq!(to_pascal_case("currentSpeed"), "CurrentSpeed");
    }

    #[test]
    fn test_enum_generation() {
        let values = vec!["green".to_string(), "yellow".to_string(), "red".to_string()];
        let options = TsOptions::default();
        let enum_def = generate_enum_type("TrafficLight", &values, &options);

        assert!(enum_def.contains("export enum TrafficLight"));
        assert!(enum_def.contains("Green = \"green\""));
        assert!(enum_def.contains("Yellow = \"yellow\""));
        assert!(enum_def.contains("Red = \"red\""));
    }

    #[test]
    fn test_ts_options_default() {
        let options = TsOptions::default();
        assert!(!options.export_default);
        assert!(options.strict_null_checks);
        assert!(!options.readonly_properties);
        assert!(options.snake_case_to_camel);
    }
}
