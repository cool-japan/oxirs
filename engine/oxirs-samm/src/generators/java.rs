//! SAMM to Java Code Generator
//!
//! Generates Java classes (POJOs, Records) from SAMM Aspect models.
//! Supports Java 8-21 features including Records, annotations, and various frameworks.

use crate::error::SammError;
use crate::metamodel::{Aspect, CharacteristicKind, ModelElement};
use std::collections::HashSet;

/// Java generation options
#[derive(Debug, Clone)]
pub struct JavaOptions {
    /// Use Java Records (Java 14+)
    pub use_records: bool,
    /// Add Jackson annotations for JSON serialization
    pub add_jackson: bool,
    /// Add Lombok annotations
    pub add_lombok: bool,
    /// Add Bean Validation (JSR-303) annotations
    pub add_validation: bool,
    /// Generate builder pattern
    pub generate_builder: bool,
    /// Package name for generated classes
    pub package_name: String,
}

impl Default for JavaOptions {
    fn default() -> Self {
        Self {
            use_records: true,
            add_jackson: true,
            add_lombok: false,
            add_validation: true,
            generate_builder: false,
            package_name: "com.example.samm".to_string(),
        }
    }
}

/// Generate Java code from SAMM Aspect
pub fn generate_java(aspect: &Aspect, options: JavaOptions) -> Result<String, SammError> {
    let mut java = String::new();
    let mut enum_classes = Vec::new();
    let mut nested_classes = HashSet::new();

    // Package declaration
    java.push_str(&format!("package {};\n\n", options.package_name));

    // Imports
    java.push_str("// Generated from SAMM Aspect Model\n");
    if !options.use_records {
        java.push_str("import java.util.Objects;\n");
    }
    java.push_str("import java.util.List;\n");
    java.push_str("import java.util.Set;\n");
    java.push_str("import java.time.LocalDate;\n");
    java.push_str("import java.time.LocalDateTime;\n");
    java.push_str("import java.time.LocalTime;\n");

    if options.add_jackson {
        java.push_str("import com.fasterxml.jackson.annotation.JsonProperty;\n");
        java.push_str("import com.fasterxml.jackson.annotation.JsonCreator;\n");
        java.push_str("import com.fasterxml.jackson.databind.annotation.JsonDeserialize;\n");
    }

    if options.add_lombok && !options.use_records {
        java.push_str("import lombok.Data;\n");
        java.push_str("import lombok.Builder;\n");
        java.push_str("import lombok.NoArgsConstructor;\n");
        java.push_str("import lombok.AllArgsConstructor;\n");
    }

    if options.add_validation {
        java.push_str("import javax.validation.constraints.NotNull;\n");
        java.push_str("import javax.validation.constraints.Size;\n");
        java.push_str("import javax.validation.constraints.Pattern;\n");
    }

    java.push('\n');

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
        java.push_str(enum_def);
        java.push_str("\n\n");
    }

    // Generate nested class stubs
    for nested_class in &nested_classes {
        java.push_str(&generate_nested_class_stub(nested_class, &options));
        java.push_str("\n\n");
    }

    // Generate main class
    let main_class = generate_main_class(aspect, &options)?;
    java.push_str(&main_class);

    Ok(java)
}

/// Generate enum class
fn generate_enum_class(name: &str, values: &[String], options: &JavaOptions) -> String {
    let mut enum_def = String::new();

    // Javadoc
    enum_def.push_str("/**\n");
    enum_def.push_str(&format!(" * Enumeration: {}\n", name));
    enum_def.push_str(" * @generated from SAMM model\n");
    enum_def.push_str(" */\n");

    enum_def.push_str(&format!("public enum {} {{\n", name));

    // Enum constants
    for (i, value) in values.iter().enumerate() {
        let member_name = to_screaming_snake_case(value);
        let comma = if i < values.len() - 1 { "," } else { ";" };
        enum_def.push_str(&format!("    {}(\"{}\"){}\n", member_name, value, comma));
    }

    enum_def.push('\n');

    // Value field
    enum_def.push_str("    private final String value;\n\n");

    // Constructor
    enum_def.push_str(&format!("    {}(String value) {{\n", name));
    enum_def.push_str("        this.value = value;\n");
    enum_def.push_str("    }\n\n");

    // Getter
    if options.add_jackson {
        enum_def.push_str("    @JsonValue\n");
    }
    enum_def.push_str("    public String getValue() {\n");
    enum_def.push_str("        return value;\n");
    enum_def.push_str("    }\n\n");

    // fromValue method
    if options.add_jackson {
        enum_def.push_str("    @JsonCreator\n");
    }
    enum_def.push_str("    public static ");
    enum_def.push_str(name);
    enum_def.push_str(" fromValue(String value) {\n");
    enum_def.push_str("        for (");
    enum_def.push_str(name);
    enum_def.push_str(" e : values()) {\n");
    enum_def.push_str("            if (e.value.equals(value)) {\n");
    enum_def.push_str("                return e;\n");
    enum_def.push_str("            }\n");
    enum_def.push_str("        }\n");
    enum_def.push_str("        throw new IllegalArgumentException(\"Unknown value: \" + value);\n");
    enum_def.push_str("    }\n");

    enum_def.push_str("}\n");

    enum_def
}

/// Generate nested class stub
fn generate_nested_class_stub(name: &str, options: &JavaOptions) -> String {
    let mut class_def = String::new();

    // Javadoc
    class_def.push_str("/**\n");
    class_def.push_str(&format!(" * Referenced entity: {}\n", name));
    class_def.push_str(" * This is a stub - implement based on your entity structure\n");
    class_def.push_str(" * @generated from SAMM model\n");
    class_def.push_str(" */\n");

    if options.use_records {
        class_def.push_str(&format!("public record {}(String id) {{}}\n", name));
    } else {
        if options.add_lombok {
            class_def.push_str("@Data\n");
        }
        class_def.push_str(&format!("public class {} {{\n", name));
        class_def.push_str("    private String id;\n");

        if !options.add_lombok {
            // Generate getter/setter
            class_def.push('\n');
            class_def.push_str("    public String getId() {\n");
            class_def.push_str("        return id;\n");
            class_def.push_str("    }\n\n");
            class_def.push_str("    public void setId(String id) {\n");
            class_def.push_str("        this.id = id;\n");
            class_def.push_str("    }\n");
        }

        class_def.push_str("}\n");
    }

    class_def
}

/// Generate main class
fn generate_main_class(aspect: &Aspect, options: &JavaOptions) -> Result<String, SammError> {
    let class_name = to_pascal_case(&aspect.name());
    let mut class_def = String::new();

    // Javadoc
    class_def.push_str("/**\n");
    class_def.push_str(&format!(" * {}\n", class_name));
    if let Some(desc) = aspect.metadata().get_description("en") {
        class_def.push_str(" * <p>\n");
        class_def.push_str(&format!(" * {}\n", desc));
    }
    class_def.push_str(" * </p>\n");
    class_def.push_str(" * @generated from SAMM model\n");
    class_def.push_str(&format!(" * @urn {}\n", aspect.metadata().urn));
    class_def.push_str(" */\n");

    if options.use_records {
        // Generate record
        class_def.push_str(&generate_record_class(aspect, &class_name, options)?);
    } else {
        // Generate traditional class
        class_def.push_str(&generate_traditional_class(aspect, &class_name, options)?);
    }

    Ok(class_def)
}

/// Generate record class (Java 14+)
fn generate_record_class(
    aspect: &Aspect,
    class_name: &str,
    options: &JavaOptions,
) -> Result<String, SammError> {
    let mut record_def = String::new();

    if options.add_jackson {
        record_def.push_str("@JsonDeserialize\n");
    }

    record_def.push_str(&format!("public record {}(\n", class_name));

    // Always add id field
    record_def.push_str("    ");
    if options.add_validation {
        record_def.push_str("@NotNull ");
    }
    if options.add_jackson {
        record_def.push_str("@JsonProperty(\"id\") ");
    }
    record_def.push_str("String id");

    // Add properties
    for prop in aspect.properties() {
        record_def.push_str(",\n    ");

        let field_name = to_camel_case(&prop.name());
        let field_type = get_java_type(prop, options)?;

        // Add validation annotations
        if options.add_validation && !prop.optional {
            record_def.push_str("@NotNull ");
        }

        // Add Jackson annotation
        if options.add_jackson {
            record_def.push_str(&format!("@JsonProperty(\"{}\") ", field_name));
        }

        record_def.push_str(&format!("{} {}", field_type, field_name));
    }

    // Add metadata fields
    record_def.push_str(",\n    ");
    if options.add_jackson {
        record_def.push_str("@JsonProperty(\"createdAt\") ");
    }
    record_def.push_str("LocalDateTime createdAt");

    record_def.push_str(",\n    ");
    if options.add_jackson {
        record_def.push_str("@JsonProperty(\"updatedAt\") ");
    }
    record_def.push_str("LocalDateTime updatedAt");

    record_def.push_str("\n) {}\n");

    Ok(record_def)
}

/// Generate traditional class (Java 8+)
fn generate_traditional_class(
    aspect: &Aspect,
    class_name: &str,
    options: &JavaOptions,
) -> Result<String, SammError> {
    let mut class_def = String::new();

    // Annotations
    if options.add_lombok {
        class_def.push_str("@Data\n");
        if options.generate_builder {
            class_def.push_str("@Builder\n");
        }
        class_def.push_str("@NoArgsConstructor\n");
        class_def.push_str("@AllArgsConstructor\n");
    }

    class_def.push_str(&format!("public class {} {{\n", class_name));

    // Fields
    class_def.push_str("    ");
    if options.add_validation {
        class_def.push_str("@NotNull\n    ");
    }
    if options.add_jackson {
        class_def.push_str("@JsonProperty(\"id\")\n    ");
    }
    class_def.push_str("private String id;\n\n");

    for prop in aspect.properties() {
        let field_name = to_camel_case(&prop.name());
        let field_type = get_java_type(prop, options)?;

        // Javadoc
        if let Some(desc) = prop.metadata().get_description("en") {
            class_def.push_str("    /**\n");
            class_def.push_str(&format!("     * {}\n", desc));
            class_def.push_str("     */\n");
        }

        // Annotations
        if options.add_validation && !prop.optional {
            class_def.push_str("    @NotNull\n");
        }
        if options.add_jackson {
            class_def.push_str(&format!("    @JsonProperty(\"{}\")\n", field_name));
        }

        class_def.push_str(&format!("    private {} {};\n\n", field_type, field_name));
    }

    // Metadata fields
    class_def.push_str("    /** Timestamp of creation */\n");
    if options.add_jackson {
        class_def.push_str("    @JsonProperty(\"createdAt\")\n");
    }
    class_def.push_str("    private LocalDateTime createdAt;\n\n");

    class_def.push_str("    /** Timestamp of last update */\n");
    if options.add_jackson {
        class_def.push_str("    @JsonProperty(\"updatedAt\")\n");
    }
    class_def.push_str("    private LocalDateTime updatedAt;\n");

    if !options.add_lombok {
        // Generate getters and setters
        class_def.push('\n');
        class_def.push_str(&generate_getters_setters(aspect, options)?);
    }

    class_def.push_str("}\n");

    Ok(class_def)
}

/// Generate getters and setters
fn generate_getters_setters(aspect: &Aspect, options: &JavaOptions) -> Result<String, SammError> {
    let mut methods = String::new();

    // ID getter/setter
    methods.push_str("    public String getId() {\n");
    methods.push_str("        return id;\n");
    methods.push_str("    }\n\n");
    methods.push_str("    public void setId(String id) {\n");
    methods.push_str("        this.id = id;\n");
    methods.push_str("    }\n\n");

    // Property getters/setters
    for prop in aspect.properties() {
        let field_name = to_camel_case(&prop.name());
        let field_type = get_java_type(prop, options)?;
        let getter_name = format!("get{}", to_pascal_case(&prop.name()));
        let setter_name = format!("set{}", to_pascal_case(&prop.name()));

        // Getter
        methods.push_str(&format!("    public {} {}() {{\n", field_type, getter_name));
        methods.push_str(&format!("        return {};\n", field_name));
        methods.push_str("    }\n\n");

        // Setter
        methods.push_str(&format!(
            "    public void {}({} {}) {{\n",
            setter_name, field_type, field_name
        ));
        methods.push_str(&format!("        this.{} = {};\n", field_name, field_name));
        methods.push_str("    }\n\n");
    }

    // Metadata getters/setters
    methods.push_str("    public LocalDateTime getCreatedAt() {\n");
    methods.push_str("        return createdAt;\n");
    methods.push_str("    }\n\n");
    methods.push_str("    public void setCreatedAt(LocalDateTime createdAt) {\n");
    methods.push_str("        this.createdAt = createdAt;\n");
    methods.push_str("    }\n\n");
    methods.push_str("    public LocalDateTime getUpdatedAt() {\n");
    methods.push_str("        return updatedAt;\n");
    methods.push_str("    }\n\n");
    methods.push_str("    public void setUpdatedAt(LocalDateTime updatedAt) {\n");
    methods.push_str("        this.updatedAt = updatedAt;\n");
    methods.push_str("    }\n");

    Ok(methods)
}

/// Get Java type for a property
fn get_java_type(
    prop: &crate::metamodel::Property,
    options: &JavaOptions,
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
                    let element_type = map_xsd_to_java(dt);
                    format!("List<{}>", element_type)
                } else {
                    "List<String>".to_string()
                }
            }
            CharacteristicKind::Set { .. } | CharacteristicKind::SortedSet { .. } => {
                // Set type
                if let Some(dt) = &char.data_type {
                    let element_type = map_xsd_to_java(dt);
                    format!("Set<{}>", element_type)
                } else {
                    "Set<String>".to_string()
                }
            }
            CharacteristicKind::SingleEntity { entity_type } => {
                // Object reference - extract entity name
                let entity_name = entity_type.split('#').next_back().unwrap_or(entity_type);
                to_pascal_case(entity_name)
            }
            CharacteristicKind::TimeSeries { .. } => {
                // Time series as list of maps
                "List<Map<String, Object>>".to_string()
            }
            _ => {
                // Default: use data type
                if let Some(dt) = &char.data_type {
                    map_xsd_to_java(dt)
                } else {
                    "String".to_string()
                }
            }
        };

        // Java doesn't have explicit optional syntax like TypeScript,
        // but we use wrapper types for optional primitives
        Ok(base_type)
    } else {
        // No characteristic, default to String
        Ok("String".to_string())
    }
}

/// Map XSD types to Java types
fn map_xsd_to_java(xsd_type: &str) -> String {
    match xsd_type {
        t if t.ends_with("string") => "String".to_string(),
        t if t.ends_with("int") | t.ends_with("integer") => "Integer".to_string(),
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
        t if t.ends_with("duration") => "Duration".to_string(),
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
    let first = chars
        .next()
        .expect("iterator should have next element")
        .to_lowercase()
        .to_string();
    format!("{}{}", first, chars.as_str())
}

/// Convert to SCREAMING_SNAKE_CASE for enum constants
fn to_screaming_snake_case(s: &str) -> String {
    s.to_uppercase().replace(['-', ' ', '.'], "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xsd_to_java_mapping() {
        assert_eq!(
            map_xsd_to_java("http://www.w3.org/2001/XMLSchema#string"),
            "String"
        );
        assert_eq!(
            map_xsd_to_java("http://www.w3.org/2001/XMLSchema#int"),
            "Integer"
        );
        assert_eq!(
            map_xsd_to_java("http://www.w3.org/2001/XMLSchema#boolean"),
            "Boolean"
        );
        assert_eq!(
            map_xsd_to_java("http://www.w3.org/2001/XMLSchema#float"),
            "Float"
        );
        assert_eq!(
            map_xsd_to_java("http://www.w3.org/2001/XMLSchema#dateTime"),
            "LocalDateTime"
        );
    }

    #[test]
    fn test_case_conversion() {
        assert_eq!(to_pascal_case("movement_aspect"), "MovementAspect");
        assert_eq!(to_pascal_case("currentSpeed"), "CurrentSpeed");
        assert_eq!(to_camel_case("MovementAspect"), "movementAspect");
        assert_eq!(to_camel_case("current_speed"), "currentSpeed");
        assert_eq!(to_screaming_snake_case("traffic-light"), "TRAFFIC_LIGHT");
    }

    #[test]
    fn test_java_options_default() {
        let options = JavaOptions::default();
        assert!(options.use_records);
        assert!(options.add_jackson);
        assert!(!options.add_lombok);
        assert!(options.add_validation);
        assert!(!options.generate_builder);
        assert_eq!(options.package_name, "com.example.samm");
    }

    #[test]
    fn test_enum_generation() {
        let values = vec!["green".to_string(), "yellow".to_string(), "red".to_string()];
        let options = JavaOptions::default();
        let enum_def = generate_enum_class("TrafficLight", &values, &options);

        assert!(enum_def.contains("public enum TrafficLight"));
        assert!(enum_def.contains("GREEN(\"green\")"));
        assert!(enum_def.contains("YELLOW(\"yellow\")"));
        assert!(enum_def.contains("RED(\"red\")"));
        assert!(enum_def.contains("public String getValue()"));
        assert!(enum_def.contains("public static TrafficLight fromValue"));
    }
}
