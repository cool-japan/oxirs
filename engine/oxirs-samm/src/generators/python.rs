//! SAMM to Python Dataclass Generator
//!
//! Generates Python dataclasses with type hints from SAMM Aspect models.
//! Supports typing module, Enum classes, and validation.

use crate::error::SammError;
use crate::metamodel::{Aspect, CharacteristicKind, ModelElement};
use std::collections::HashSet;

/// Python generation options
#[derive(Debug, Clone)]
pub struct PythonOptions {
    /// Use pydantic for validation
    pub use_pydantic: bool,
    /// Add __post_init__ validation
    pub add_validation: bool,
    /// Generate docstrings
    pub generate_docstrings: bool,
    /// Use snake_case for field names
    pub snake_case_fields: bool,
}

impl Default for PythonOptions {
    fn default() -> Self {
        Self {
            use_pydantic: false,
            add_validation: true,
            generate_docstrings: true,
            snake_case_fields: true,
        }
    }
}

/// Generate Python dataclasses from SAMM Aspect
pub fn generate_python(aspect: &Aspect, options: PythonOptions) -> Result<String, SammError> {
    let mut python = String::new();
    let mut enum_classes = Vec::new();
    let mut nested_classes = HashSet::new();

    // Header imports
    python.push_str("\"\"\"Generated Python dataclasses from SAMM model\"\"\"\n\n");

    if options.use_pydantic {
        python.push_str("from pydantic import BaseModel, Field\n");
    } else {
        python.push_str("from dataclasses import dataclass, field\n");
    }

    python.push_str("from typing import Optional, List\n");
    python.push_str("from enum import Enum\n");
    python.push_str("from datetime import datetime, date, time\n");
    python.push('\n');

    // Collect enum classes and nested types
    for prop in aspect.properties() {
        if let Some(char) = &prop.characteristic {
            // Collect enum classes
            if let CharacteristicKind::Enumeration { values } = char.kind() {
                let enum_name = to_pascal_case(&prop.name());
                let enum_def = generate_enum_class(&enum_name, values, &options);
                enum_classes.push(enum_def);
            }

            // Collect state classes
            if let CharacteristicKind::State {
                values,
                default_value: _,
            } = char.kind()
            {
                let enum_name = to_pascal_case(&prop.name());
                let enum_def = generate_enum_class(&enum_name, values, &options);
                enum_classes.push(enum_def);
            }

            // Collect entity types
            if let CharacteristicKind::SingleEntity { entity_type } = char.kind() {
                let entity_name = entity_type.split('#').next_back().unwrap_or(entity_type);
                nested_classes.insert(to_pascal_case(entity_name));
            }
        }
    }

    // Generate enum classes
    for enum_def in &enum_classes {
        python.push_str(enum_def);
        python.push_str("\n\n");
    }

    // Generate nested class stubs
    for nested_class in &nested_classes {
        python.push_str(&generate_nested_class_stub(nested_class, &options));
        python.push_str("\n\n");
    }

    // Generate main dataclass
    let main_class = generate_main_dataclass(aspect, &options)?;
    python.push_str(&main_class);

    Ok(python)
}

/// Generate enum class
fn generate_enum_class(name: &str, values: &[String], options: &PythonOptions) -> String {
    let mut enum_def = String::new();

    if options.generate_docstrings {
        enum_def.push_str(&format!("class {}(Enum):\n", name));
        enum_def.push_str(&format!("    \"\"\"Enumeration: {}\"\"\"\n", name));
    } else {
        enum_def.push_str(&format!("class {}(Enum):\n", name));
    }

    for value in values {
        let member_name = value.to_uppercase().replace(['-', ' '], "_");
        enum_def.push_str(&format!("    {} = \"{}\"\n", member_name, value));
    }

    enum_def
}

/// Generate nested class stub
fn generate_nested_class_stub(name: &str, options: &PythonOptions) -> String {
    let mut class_def = String::new();

    if options.use_pydantic {
        class_def.push_str(&format!("class {}(BaseModel):\n", name));
    } else {
        class_def.push_str("@dataclass\n");
        class_def.push_str(&format!("class {}:\n", name));
    }

    if options.generate_docstrings {
        class_def.push_str(&format!("    \"\"\"Referenced entity: {}\"\"\"\n", name));
    }

    class_def.push_str("    id: str\n");
    class_def.push_str("    # Add other fields as needed\n");

    class_def
}

/// Generate main dataclass
fn generate_main_dataclass(aspect: &Aspect, options: &PythonOptions) -> Result<String, SammError> {
    let class_name = to_pascal_case(&aspect.name());
    let mut class_def = String::new();

    // Decorator and class definition
    if options.use_pydantic {
        class_def.push_str(&format!("class {}(BaseModel):\n", class_name));
    } else {
        class_def.push_str("@dataclass\n");
        class_def.push_str(&format!("class {}:\n", class_name));
    }

    // Docstring
    if options.generate_docstrings {
        class_def.push_str(&format!("    \"\"\"{}\n", class_name));
        if let Some(desc) = aspect.metadata().get_description("en") {
            class_def.push_str(&format!("    \n    {}\n", desc));
        }
        class_def.push_str("    \n    Generated from SAMM model\n");
        class_def.push_str(&format!("    URN: {}\n", aspect.metadata().urn));
        class_def.push_str("    \"\"\"\n");
    }

    // Always add id field
    class_def.push_str("    id: str\n");

    // Add properties
    for prop in aspect.properties() {
        let field_name = if options.snake_case_fields {
            to_snake_case(&prop.name())
        } else {
            prop.name().to_string()
        };

        let field_type = get_python_type(prop, options)?;

        // Add docstring comment
        if options.generate_docstrings {
            if let Some(desc) = prop.metadata().get_description("en") {
                class_def.push_str(&format!("    # {}\n", desc));
            }
        }

        class_def.push_str(&format!("    {}: {}\n", field_name, field_type));
    }

    // Add metadata fields
    class_def.push_str("    created_at: datetime\n");
    class_def.push_str("    updated_at: datetime\n");

    // Add __post_init__ validation if requested
    if options.add_validation && !options.use_pydantic {
        class_def.push('\n');
        class_def.push_str("    def __post_init__(self):\n");
        class_def.push_str("        \"\"\"Validate field values\"\"\"\n");

        // Add validation for required fields
        for prop in aspect.properties() {
            if !prop.optional {
                let field_name = if options.snake_case_fields {
                    to_snake_case(&prop.name())
                } else {
                    prop.name().to_string()
                };
                class_def.push_str(&format!("        if self.{} is None:\n", field_name));
                class_def.push_str(&format!(
                    "            raise ValueError(\"Field '{}' is required\")\n",
                    field_name
                ));
            }
        }
    }

    Ok(class_def)
}

/// Get Python type for a property
fn get_python_type(
    prop: &crate::metamodel::Property,
    options: &PythonOptions,
) -> Result<String, SammError> {
    if let Some(char) = &prop.characteristic {
        let base_type = match char.kind() {
            CharacteristicKind::Enumeration { .. } | CharacteristicKind::State { .. } => {
                // Use the property name as the enum class name
                to_pascal_case(&prop.name())
            }
            CharacteristicKind::Collection { .. } | CharacteristicKind::List { .. } => {
                // List type
                if let Some(dt) = &char.data_type {
                    let element_type = map_xsd_to_python(dt);
                    format!("List[{}]", element_type)
                } else {
                    "List[str]".to_string()
                }
            }
            CharacteristicKind::Set { .. } | CharacteristicKind::SortedSet { .. } => {
                // Set type (represented as List in Python for JSON serialization)
                if let Some(dt) = &char.data_type {
                    let element_type = map_xsd_to_python(dt);
                    format!("List[{}]", element_type)
                } else {
                    "List[str]".to_string()
                }
            }
            CharacteristicKind::SingleEntity { entity_type } => {
                // Object reference - extract entity name
                let entity_name = entity_type.split('#').next_back().unwrap_or(entity_type);
                to_pascal_case(entity_name)
            }
            CharacteristicKind::TimeSeries { .. } => {
                // Time series as list of dicts
                "List[dict]".to_string()
            }
            _ => {
                // Default: use data type
                if let Some(dt) = &char.data_type {
                    map_xsd_to_python(dt)
                } else {
                    "str".to_string()
                }
            }
        };

        // Add Optional for optional fields
        if prop.optional {
            Ok(format!("Optional[{}]", base_type))
        } else {
            Ok(base_type)
        }
    } else {
        // No characteristic, default to str
        if prop.optional {
            Ok("Optional[str]".to_string())
        } else {
            Ok("str".to_string())
        }
    }
}

/// Map XSD types to Python types
fn map_xsd_to_python(xsd_type: &str) -> String {
    match xsd_type {
        t if t.ends_with("string") => "str".to_string(),
        t if t.ends_with("int") | t.ends_with("integer") => "int".to_string(),
        t if t.ends_with("long") => "int".to_string(),
        t if t.ends_with("short") | t.ends_with("byte") => "int".to_string(),
        t if t.ends_with("decimal") => "float".to_string(),
        t if t.ends_with("float") => "float".to_string(),
        t if t.ends_with("double") => "float".to_string(),
        t if t.ends_with("boolean") => "bool".to_string(),
        t if t.ends_with("date") => "date".to_string(),
        t if t.ends_with("dateTime") | t.ends_with("dateTimeStamp") => "datetime".to_string(),
        t if t.ends_with("time") => "time".to_string(),
        t if t.ends_with("duration") => "str".to_string(),
        t if t.ends_with("anyURI") => "str".to_string(),
        _ => "str".to_string(),
    }
}

/// Convert PascalCase/camelCase to snake_case
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(
                ch.to_lowercase()
                    .next()
                    .expect("lowercase should produce a character"),
            );
        } else {
            result.push(ch);
        }
    }
    result
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
    fn test_xsd_to_python_mapping() {
        assert_eq!(
            map_xsd_to_python("http://www.w3.org/2001/XMLSchema#string"),
            "str"
        );
        assert_eq!(
            map_xsd_to_python("http://www.w3.org/2001/XMLSchema#int"),
            "int"
        );
        assert_eq!(
            map_xsd_to_python("http://www.w3.org/2001/XMLSchema#boolean"),
            "bool"
        );
        assert_eq!(
            map_xsd_to_python("http://www.w3.org/2001/XMLSchema#float"),
            "float"
        );
        assert_eq!(
            map_xsd_to_python("http://www.w3.org/2001/XMLSchema#dateTime"),
            "datetime"
        );
    }

    #[test]
    fn test_case_conversion() {
        assert_eq!(to_snake_case("MovementAspect"), "movement_aspect");
        assert_eq!(to_snake_case("currentSpeed"), "current_speed");
        assert_eq!(to_pascal_case("movement_aspect"), "MovementAspect");
        assert_eq!(to_pascal_case("currentSpeed"), "CurrentSpeed");
    }

    #[test]
    fn test_python_options_default() {
        let options = PythonOptions::default();
        assert!(!options.use_pydantic);
        assert!(options.add_validation);
        assert!(options.generate_docstrings);
        assert!(options.snake_case_fields);
    }
}
