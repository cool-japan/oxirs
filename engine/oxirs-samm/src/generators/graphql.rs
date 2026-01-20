//! SAMM to GraphQL Schema Generator
//!
//! Generates GraphQL schema definitions from SAMM Aspect models.
//! Supports type mapping, enumerations, collections, and entity relationships.

use crate::error::SammError;
use crate::metamodel::{Aspect, CharacteristicKind, ModelElement};
use std::collections::HashSet;

/// Generate GraphQL schema from SAMM Aspect
pub fn generate_graphql(aspect: &Aspect) -> Result<String, SammError> {
    let mut schema = String::new();
    let mut custom_scalars = HashSet::new();
    let mut enum_types = Vec::new();

    // Header comment
    schema.push_str(&format!("# GraphQL schema for {}\n", aspect.name()));
    schema.push_str("# Generated from SAMM Aspect Model\n");
    schema.push_str(&format!("# URN: {}\n\n", aspect.metadata().urn));

    // Collect custom scalars and types
    for prop in aspect.properties() {
        if let Some(char) = &prop.characteristic {
            // Check for custom scalars needed
            if let Some(dt) = &char.data_type {
                if let Some(scalar) = get_custom_scalar_for_xsd(dt) {
                    custom_scalars.insert(scalar);
                }
            }

            // Collect enum types
            if let CharacteristicKind::Enumeration { values } = char.kind() {
                let enum_name = to_pascal_case(&prop.name());
                let enum_def = generate_enum_type(&enum_name, values);
                enum_types.push(enum_def);
            }

            // Collect state types
            if let CharacteristicKind::State {
                values,
                default_value: _,
            } = char.kind()
            {
                let enum_name = to_pascal_case(&prop.name());
                let enum_def = generate_enum_type(&enum_name, values);
                enum_types.push(enum_def);
            }
        }
    }

    // Generate custom scalar declarations
    for scalar in &custom_scalars {
        schema.push_str(&format!("scalar {}\n", scalar));
    }
    if !custom_scalars.is_empty() {
        schema.push('\n');
    }

    // Generate enum types
    for enum_def in &enum_types {
        schema.push_str(enum_def);
        schema.push('\n');
    }

    // Generate main type
    let main_type = generate_main_type(aspect)?;
    schema.push_str(&main_type);
    schema.push('\n');

    // Generate Query type
    let query_type = generate_query_type(aspect)?;
    schema.push_str(&query_type);

    Ok(schema)
}

/// Generate enum type definition
fn generate_enum_type(name: &str, values: &[String]) -> String {
    let mut enum_def = String::new();
    enum_def.push_str(&format!("enum {} {{\n", name));

    for value in values {
        // Convert to valid GraphQL enum value (uppercase with underscores)
        let enum_value = value.to_uppercase().replace(['-', ' '], "_");
        enum_def.push_str(&format!("  {}\n", enum_value));
    }

    enum_def.push_str("}\n");
    enum_def
}

/// Generate main object type
fn generate_main_type(aspect: &Aspect) -> Result<String, SammError> {
    let type_name = to_pascal_case(&aspect.name());
    let mut type_def = String::new();

    // Add description if available
    if let Some(desc) = aspect.metadata().get_description("en") {
        type_def.push_str(&format!("\"\"\"\n{}\n\"\"\"\n", desc));
    }

    type_def.push_str(&format!("type {} {{\n", type_name));

    // Always add ID field
    type_def.push_str("  \"\"\"Unique identifier\"\"\"\n");
    type_def.push_str("  id: ID!\n\n");

    // Add properties
    for prop in aspect.properties() {
        let field_name = to_camel_case(&prop.name());
        let field_type = get_graphql_type(prop)?;

        // Add description if available
        if let Some(desc) = prop.metadata().get_description("en") {
            type_def.push_str(&format!("  \"\"\"\n  {}\n  \"\"\"\n", desc));
        }

        type_def.push_str(&format!("  {}: {}\n\n", field_name, field_type));
    }

    // Add metadata fields
    type_def.push_str("  \"\"\"Timestamp of creation\"\"\"\n");
    type_def.push_str("  createdAt: DateTime!\n\n");
    type_def.push_str("  \"\"\"Timestamp of last update\"\"\"\n");
    type_def.push_str("  updatedAt: DateTime!\n");

    type_def.push_str("}\n");
    Ok(type_def)
}

/// Generate Query type
fn generate_query_type(aspect: &Aspect) -> Result<String, SammError> {
    let type_name = to_pascal_case(&aspect.name());
    let query_name = to_camel_case(&aspect.name());
    let query_name_plural = format!("{}s", query_name);

    let mut query_def = String::new();
    query_def.push_str("type Query {\n");
    query_def.push_str(&format!("  \"\"\"Get a single {} by ID\"\"\"\n", type_name));
    query_def.push_str(&format!("  {}(id: ID!): {}\n\n", query_name, type_name));
    query_def.push_str(&format!(
        "  \"\"\"Get multiple {}s with pagination\"\"\"\n",
        type_name
    ));
    query_def.push_str(&format!(
        "  {}(limit: Int = 10, offset: Int = 0): [{}!]!\n",
        query_name_plural, type_name
    ));
    query_def.push_str("}\n");

    Ok(query_def)
}

/// Get GraphQL type for a property
fn get_graphql_type(prop: &crate::metamodel::Property) -> Result<String, SammError> {
    if let Some(char) = &prop.characteristic {
        let base_type = match char.kind() {
            CharacteristicKind::Enumeration { .. } | CharacteristicKind::State { .. } => {
                // Use the property name as the enum type name
                to_pascal_case(&prop.name())
            }
            CharacteristicKind::Collection { .. } | CharacteristicKind::List { .. } => {
                // Array type
                if let Some(dt) = &char.data_type {
                    let element_type = map_xsd_to_graphql(dt);
                    format!("[{}!]", element_type)
                } else {
                    "[String!]".to_string()
                }
            }
            CharacteristicKind::SingleEntity { entity_type } => {
                // Object reference - extract entity name
                let entity_name = entity_type.split('#').next_back().unwrap_or(entity_type);
                to_pascal_case(entity_name)
            }
            CharacteristicKind::Set { .. } => {
                // Set is similar to collection
                if let Some(dt) = &char.data_type {
                    let element_type = map_xsd_to_graphql(dt);
                    format!("[{}!]", element_type)
                } else {
                    "[String!]".to_string()
                }
            }
            CharacteristicKind::SortedSet { .. } => {
                // Sorted set
                if let Some(dt) = &char.data_type {
                    let element_type = map_xsd_to_graphql(dt);
                    format!("[{}!]", element_type)
                } else {
                    "[String!]".to_string()
                }
            }
            CharacteristicKind::TimeSeries { .. } => {
                // Time series as array of timestamped values
                "[TimeSeriesValue!]".to_string()
            }
            _ => {
                // Default: use data type
                if let Some(dt) = &char.data_type {
                    map_xsd_to_graphql(dt)
                } else {
                    "String".to_string()
                }
            }
        };

        // Add non-null modifier if not optional
        if prop.optional {
            Ok(base_type)
        } else {
            // Check if it's already an array type
            if base_type.starts_with('[') {
                Ok(format!("{}!", base_type))
            } else if base_type.ends_with('!') {
                Ok(base_type)
            } else {
                Ok(format!("{}!", base_type))
            }
        }
    } else {
        // No characteristic, default to String
        if prop.optional {
            Ok("String".to_string())
        } else {
            Ok("String!".to_string())
        }
    }
}

/// Map XSD types to GraphQL types
fn map_xsd_to_graphql(xsd_type: &str) -> String {
    match xsd_type {
        t if t.ends_with("string") => "String".to_string(),
        t if t.ends_with("int") | t.ends_with("integer") => "Int".to_string(),
        t if t.ends_with("long") => "Int".to_string(),
        t if t.ends_with("short") | t.ends_with("byte") => "Int".to_string(),
        t if t.ends_with("decimal") => "Float".to_string(),
        t if t.ends_with("float") => "Float".to_string(),
        t if t.ends_with("double") => "Float".to_string(),
        t if t.ends_with("boolean") => "Boolean".to_string(),
        t if t.ends_with("date") => "Date".to_string(),
        t if t.ends_with("dateTime") | t.ends_with("dateTimeStamp") => "DateTime".to_string(),
        t if t.ends_with("time") => "Time".to_string(),
        t if t.ends_with("duration") => "Duration".to_string(),
        t if t.ends_with("anyURI") => "String".to_string(),
        t if t.ends_with("base64Binary") => "String".to_string(),
        t if t.ends_with("hexBinary") => "String".to_string(),
        _ => "String".to_string(),
    }
}

/// Get custom scalar name if needed for XSD type
fn get_custom_scalar_for_xsd(xsd_type: &str) -> Option<String> {
    match xsd_type {
        t if t.ends_with("date") => Some("Date".to_string()),
        t if t.ends_with("dateTime") | t.ends_with("dateTimeStamp") => Some("DateTime".to_string()),
        t if t.ends_with("time") => Some("Time".to_string()),
        t if t.ends_with("duration") => Some("Duration".to_string()),
        _ => None,
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
        if ch == '_' || ch == '-' {
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
    fn test_xsd_to_graphql_mapping() {
        assert_eq!(
            map_xsd_to_graphql("http://www.w3.org/2001/XMLSchema#string"),
            "String"
        );
        assert_eq!(
            map_xsd_to_graphql("http://www.w3.org/2001/XMLSchema#int"),
            "Int"
        );
        assert_eq!(
            map_xsd_to_graphql("http://www.w3.org/2001/XMLSchema#boolean"),
            "Boolean"
        );
        assert_eq!(
            map_xsd_to_graphql("http://www.w3.org/2001/XMLSchema#float"),
            "Float"
        );
        assert_eq!(
            map_xsd_to_graphql("http://www.w3.org/2001/XMLSchema#dateTime"),
            "DateTime"
        );
    }

    #[test]
    fn test_custom_scalar_detection() {
        assert_eq!(
            get_custom_scalar_for_xsd("http://www.w3.org/2001/XMLSchema#dateTime"),
            Some("DateTime".to_string())
        );
        assert_eq!(
            get_custom_scalar_for_xsd("http://www.w3.org/2001/XMLSchema#date"),
            Some("Date".to_string())
        );
        assert_eq!(
            get_custom_scalar_for_xsd("http://www.w3.org/2001/XMLSchema#string"),
            None
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
        let values = vec!["GREEN".to_string(), "YELLOW".to_string(), "RED".to_string()];
        let enum_def = generate_enum_type("TrafficLight", &values);

        assert!(enum_def.contains("enum TrafficLight"));
        assert!(enum_def.contains("GREEN"));
        assert!(enum_def.contains("YELLOW"));
        assert!(enum_def.contains("RED"));
    }
}
