//! SAMM to Scala Code Generator
//!
//! Generates Scala case classes from SAMM Aspect models.
//! Supports Scala 2.13 and Scala 3, with options for Play JSON, Circe, and other frameworks.

use crate::error::SammError;
use crate::metamodel::{Aspect, CharacteristicKind, ModelElement};
use std::collections::HashSet;

/// Scala generation options
#[derive(Debug, Clone)]
pub struct ScalaOptions {
    /// Use Scala 3 syntax (enums, givens)
    pub scala3: bool,
    /// Add Play JSON format annotations
    pub add_play_json: bool,
    /// Add Circe codec derivation
    pub add_circe: bool,
    /// Generate companion objects with apply methods
    pub generate_companion: bool,
    /// Package name for generated classes
    pub package_name: String,
}

impl Default for ScalaOptions {
    fn default() -> Self {
        Self {
            scala3: true,
            add_play_json: false,
            add_circe: true,
            generate_companion: true,
            package_name: "com.example.samm".to_string(),
        }
    }
}

/// Generate Scala code from SAMM Aspect
pub fn generate_scala(aspect: &Aspect, options: ScalaOptions) -> Result<String, SammError> {
    let mut scala = String::new();
    let mut enum_classes = Vec::new();
    let mut nested_classes = HashSet::new();

    // Package declaration
    scala.push_str(&format!("package {}\n\n", options.package_name));

    // Imports
    scala.push_str("// Generated from SAMM Aspect Model\n");
    scala.push_str("import java.time.{LocalDate, LocalDateTime, LocalTime}\n");

    if options.add_play_json {
        scala.push_str("import play.api.libs.json._\n");
        scala.push_str("import play.api.libs.functional.syntax._\n");
    }

    if options.add_circe {
        scala.push_str("import io.circe.{Decoder, Encoder}\n");
        scala.push_str("import io.circe.generic.semiauto._\n");
    }

    scala.push('\n');

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
        scala.push_str(enum_def);
        scala.push_str("\n\n");
    }

    // Generate nested class stubs
    for nested_class in &nested_classes {
        scala.push_str(&generate_nested_class_stub(nested_class, &options));
        scala.push_str("\n\n");
    }

    // Generate main case class
    let main_class = generate_main_case_class(aspect, &options)?;
    scala.push_str(&main_class);

    Ok(scala)
}

/// Generate enum class
fn generate_enum_class(name: &str, values: &[String], options: &ScalaOptions) -> String {
    let mut enum_def = String::new();

    if options.scala3 {
        // Scala 3 enum syntax
        enum_def.push_str(&format!("/** Enumeration: {} */\n", name));
        enum_def.push_str(&format!("enum {}:\n", name));
        for value in values {
            let case_name = to_pascal_case(value);
            enum_def.push_str(&format!("  case {}\n", case_name));
        }

        // Add companion object for Play JSON or Circe
        if options.add_play_json || options.add_circe {
            enum_def.push('\n');
            enum_def.push_str(&format!("object {}:\n", name));

            if options.add_play_json {
                enum_def.push_str("  implicit val format: Format[");
                enum_def.push_str(name);
                enum_def.push_str("] = new Format[");
                enum_def.push_str(name);
                enum_def.push_str("] {\n");
                enum_def.push_str("    def reads(json: JsValue): JsResult[");
                enum_def.push_str(name);
                enum_def.push_str("] = json.validate[String].flatMap {\n");
                enum_def.push_str("      case ");
                for (i, value) in values.iter().enumerate() {
                    if i > 0 {
                        enum_def.push_str(" | ");
                    }
                    enum_def.push_str(&format!("\"{}\"", value));
                }
                enum_def.push_str(" => JsSuccess(");
                enum_def.push_str(name);
                enum_def.push_str(&format!(".{})\n", to_pascal_case(&values[0])));
                enum_def.push_str("      case other => JsError(s\"Invalid value: $other\")\n");
                enum_def.push_str("    }\n");
                enum_def.push_str("    def writes(o: ");
                enum_def.push_str(name);
                enum_def.push_str("): JsValue = JsString(o.toString.toLowerCase)\n");
                enum_def.push_str("  }\n");
            }

            if options.add_circe {
                enum_def.push_str("  implicit val decoder: Decoder[");
                enum_def.push_str(name);
                enum_def.push_str("] = deriveDecoder\n");
                enum_def.push_str("  implicit val encoder: Encoder[");
                enum_def.push_str(name);
                enum_def.push_str("] = deriveEncoder\n");
            }
        }
    } else {
        // Scala 2 sealed trait + case objects
        enum_def.push_str(&format!("/** Enumeration: {} */\n", name));
        enum_def.push_str(&format!("sealed trait {}\n", name));
        enum_def.push_str(&format!("object {} {{\n", name));

        for value in values {
            let case_name = to_pascal_case(value);
            enum_def.push_str(&format!("  case object {} extends {}\n", case_name, name));
        }

        enum_def.push('\n');
        enum_def.push_str("  val values: Seq[");
        enum_def.push_str(name);
        enum_def.push_str("] = Seq(");
        for (i, value) in values.iter().enumerate() {
            if i > 0 {
                enum_def.push_str(", ");
            }
            enum_def.push_str(&to_pascal_case(value));
        }
        enum_def.push_str(")\n\n");

        // Play JSON format for Scala 2
        if options.add_play_json {
            enum_def.push_str("  implicit val format: Format[");
            enum_def.push_str(name);
            enum_def.push_str("] = new Format[");
            enum_def.push_str(name);
            enum_def.push_str("] {\n");
            enum_def.push_str("    def reads(json: JsValue): JsResult[");
            enum_def.push_str(name);
            enum_def.push_str("] = json.validate[String].flatMap {\n");
            for value in values {
                let case_name = to_pascal_case(value);
                enum_def.push_str(&format!(
                    "      case \"{}\" => JsSuccess({})\n",
                    value, case_name
                ));
            }
            enum_def.push_str("      case other => JsError(s\"Unknown value: $other\")\n");
            enum_def.push_str("    }\n");
            enum_def.push_str("    def writes(o: ");
            enum_def.push_str(name);
            enum_def.push_str("): JsValue = JsString(o.toString.toLowerCase)\n");
            enum_def.push_str("  }\n");
        }

        enum_def.push_str("}\n");
    }

    enum_def
}

/// Generate nested class stub
fn generate_nested_class_stub(name: &str, options: &ScalaOptions) -> String {
    let mut class_def = String::new();

    // Scaladoc
    class_def.push_str("/**\n");
    class_def.push_str(&format!(" * Referenced entity: {}\n", name));
    class_def.push_str(" * This is a stub - implement based on your entity structure\n");
    class_def.push_str(" * @generated from SAMM model\n");
    class_def.push_str(" */\n");

    class_def.push_str(&format!("case class {}(id: String)\n", name));

    // Companion object with codecs
    if options.generate_companion && (options.add_play_json || options.add_circe) {
        class_def.push_str(&format!("\nobject {} {{\n", name));

        if options.add_play_json {
            class_def.push_str("  implicit val format: Format[");
            class_def.push_str(name);
            class_def.push_str("] = Json.format[");
            class_def.push_str(name);
            class_def.push_str("]\n");
        }

        if options.add_circe {
            class_def.push_str("  implicit val decoder: Decoder[");
            class_def.push_str(name);
            class_def.push_str("] = deriveDecoder\n");
            class_def.push_str("  implicit val encoder: Encoder[");
            class_def.push_str(name);
            class_def.push_str("] = deriveEncoder\n");
        }

        class_def.push_str("}\n");
    }

    class_def
}

/// Generate main case class
fn generate_main_case_class(aspect: &Aspect, options: &ScalaOptions) -> Result<String, SammError> {
    let class_name = to_pascal_case(&aspect.name());
    let mut class_def = String::new();

    // Scaladoc
    class_def.push_str("/**\n");
    class_def.push_str(&format!(" * {}\n", class_name));
    if let Some(desc) = aspect.metadata().get_description("en") {
        class_def.push_str(" *\n");
        class_def.push_str(&format!(" * {}\n", desc));
    }
    class_def.push_str(" *\n");
    class_def.push_str(" * @generated from SAMM model\n");
    class_def.push_str(&format!(" * @urn {}\n", aspect.metadata().urn));
    class_def.push_str(" */\n");

    // Case class definition
    class_def.push_str(&format!("case class {}(\n", class_name));

    // Always add id field
    class_def.push_str("  id: String");

    // Add properties
    for prop in aspect.properties() {
        class_def.push_str(",\n  ");

        let field_name = to_camel_case(&prop.name());
        let field_type = get_scala_type(prop, options)?;

        // Add Scaladoc inline comment
        if let Some(desc) = prop.metadata().get_description("en") {
            class_def.push_str(&format!("{}: {} // {}", field_name, field_type, desc));
        } else {
            class_def.push_str(&format!("{}: {}", field_name, field_type));
        }
    }

    // Add metadata fields
    class_def.push_str(",\n  createdAt: LocalDateTime");
    class_def.push_str(",\n  updatedAt: LocalDateTime");

    class_def.push_str("\n)\n");

    // Companion object
    if options.generate_companion {
        class_def.push('\n');
        class_def.push_str(&format!("object {} {{\n", class_name));

        if options.add_play_json {
            class_def.push_str("  implicit val format: Format[");
            class_def.push_str(&class_name);
            class_def.push_str("] = Json.format[");
            class_def.push_str(&class_name);
            class_def.push_str("]\n");
        }

        if options.add_circe {
            class_def.push_str("  implicit val decoder: Decoder[");
            class_def.push_str(&class_name);
            class_def.push_str("] = deriveDecoder\n");
            class_def.push_str("  implicit val encoder: Encoder[");
            class_def.push_str(&class_name);
            class_def.push_str("] = deriveEncoder\n");
        }

        class_def.push_str("}\n");
    }

    Ok(class_def)
}

/// Get Scala type for a property
fn get_scala_type(
    prop: &crate::metamodel::Property,
    options: &ScalaOptions,
) -> Result<String, SammError> {
    if let Some(char) = &prop.characteristic {
        let base_type = match char.kind() {
            CharacteristicKind::Enumeration { .. } | CharacteristicKind::State { .. } => {
                // Use the property name as the enum type name
                to_pascal_case(&prop.name())
            }
            CharacteristicKind::Collection { .. } | CharacteristicKind::List { .. } => {
                // Seq type (Scala's default sequence)
                if let Some(dt) = &char.data_type {
                    let element_type = map_xsd_to_scala(dt);
                    format!("Seq[{}]", element_type)
                } else {
                    "Seq[String]".to_string()
                }
            }
            CharacteristicKind::Set { .. } | CharacteristicKind::SortedSet { .. } => {
                // Set type
                if let Some(dt) = &char.data_type {
                    let element_type = map_xsd_to_scala(dt);
                    format!("Set[{}]", element_type)
                } else {
                    "Set[String]".to_string()
                }
            }
            CharacteristicKind::SingleEntity { entity_type } => {
                // Object reference - extract entity name
                let entity_name = entity_type.split('#').next_back().unwrap_or(entity_type);
                to_pascal_case(entity_name)
            }
            CharacteristicKind::TimeSeries { .. } => {
                // Time series as sequence of tuples
                "Seq[(LocalDateTime, Double)]".to_string()
            }
            _ => {
                // Default: use data type
                if let Some(dt) = &char.data_type {
                    map_xsd_to_scala(dt)
                } else {
                    "String".to_string()
                }
            }
        };

        // Wrap in Option for optional fields
        if prop.optional {
            Ok(format!("Option[{}]", base_type))
        } else {
            Ok(base_type)
        }
    } else {
        // No characteristic, default to String
        if prop.optional {
            Ok("Option[String]".to_string())
        } else {
            Ok("String".to_string())
        }
    }
}

/// Map XSD types to Scala types
fn map_xsd_to_scala(xsd_type: &str) -> String {
    match xsd_type {
        t if t.ends_with("string") => "String".to_string(),
        t if t.ends_with("int") | t.ends_with("integer") => "Int".to_string(),
        t if t.ends_with("long") => "Long".to_string(),
        t if t.ends_with("short") => "Short".to_string(),
        t if t.ends_with("byte") => "Byte".to_string(),
        t if t.ends_with("decimal") => "BigDecimal".to_string(),
        t if t.ends_with("float") => "Float".to_string(),
        t if t.ends_with("double") => "Double".to_string(),
        t if t.ends_with("boolean") => "Boolean".to_string(),
        t if t.ends_with("date") => "LocalDate".to_string(),
        t if t.ends_with("dateTime") | t.ends_with("dateTimeStamp") => "LocalDateTime".to_string(),
        t if t.ends_with("time") => "LocalTime".to_string(),
        t if t.ends_with("duration") => "java.time.Duration".to_string(),
        t if t.ends_with("anyURI") => "String".to_string(),
        _ => "String".to_string(),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xsd_to_scala_mapping() {
        assert_eq!(
            map_xsd_to_scala("http://www.w3.org/2001/XMLSchema#string"),
            "String"
        );
        assert_eq!(
            map_xsd_to_scala("http://www.w3.org/2001/XMLSchema#int"),
            "Int"
        );
        assert_eq!(
            map_xsd_to_scala("http://www.w3.org/2001/XMLSchema#boolean"),
            "Boolean"
        );
        assert_eq!(
            map_xsd_to_scala("http://www.w3.org/2001/XMLSchema#float"),
            "Float"
        );
        assert_eq!(
            map_xsd_to_scala("http://www.w3.org/2001/XMLSchema#dateTime"),
            "LocalDateTime"
        );
    }

    #[test]
    fn test_case_conversion() {
        assert_eq!(to_pascal_case("movement_aspect"), "MovementAspect");
        assert_eq!(to_pascal_case("currentSpeed"), "CurrentSpeed");
        assert_eq!(to_camel_case("MovementAspect"), "movementAspect");
        assert_eq!(to_camel_case("current_speed"), "currentSpeed");
    }

    #[test]
    fn test_scala_options_default() {
        let options = ScalaOptions::default();
        assert!(options.scala3);
        assert!(!options.add_play_json);
        assert!(options.add_circe);
        assert!(options.generate_companion);
        assert_eq!(options.package_name, "com.example.samm");
    }
}
