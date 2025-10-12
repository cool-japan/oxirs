//! Template engine for custom code generation
//!
//! This module provides a flexible template system using Tera for generating
//! code from SAMM Aspect Models. It supports both built-in templates and custom
//! user-provided templates.
//!
//! ## Features
//!
//! - Built-in templates for common target languages
//! - Custom template loading from files or directories
//! - Context building from Aspect Models
//! - Template inheritance and includes
//! - Custom filters and functions
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxirs_samm::templates::{TemplateEngine, TemplateContext};
//! use oxirs_samm::metamodel::Aspect;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut engine = TemplateEngine::new()?;
//!
//! // Load custom template
//! engine.load_template_file("custom.tera")?;
//!
//! // Build context from aspect
//! let aspect = Aspect::new("MyAspect".to_string());
//! let context = TemplateContext::from_aspect(&aspect);
//!
//! // Render template
//! let output = engine.render("custom.tera", &context)?;
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, SammError};
use crate::metamodel::{Aspect, Characteristic, Entity, ModelElement, Operation, Property};
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tera::{Context, Tera, Value};

/// Template engine for SAMM code generation
pub struct TemplateEngine {
    /// Tera template engine instance
    tera: Tera,
    /// Loaded template paths for tracking
    loaded_templates: Vec<PathBuf>,
}

impl TemplateEngine {
    /// Create a new template engine with built-in templates
    pub fn new() -> Result<Self> {
        let mut tera = Tera::default();

        // Add built-in templates (embedded in binary)
        tera.add_raw_templates(vec![
            ("rust.tera", include_str!("builtin/rust.tera")),
            ("python.tera", include_str!("builtin/python.tera")),
            ("typescript.tera", include_str!("builtin/typescript.tera")),
            ("java.tera", include_str!("builtin/java.tera")),
            ("graphql.tera", include_str!("builtin/graphql.tera")),
        ])
        .map_err(|e| SammError::ParseError(format!("Failed to load built-in templates: {}", e)))?;

        // Register custom filters
        tera.register_filter("snake_case", filters::snake_case);
        tera.register_filter("camel_case", filters::camel_case);
        tera.register_filter("pascal_case", filters::pascal_case);
        tera.register_filter("kebab_case", filters::kebab_case);
        tera.register_filter("upper_case", filters::upper_case);
        tera.register_filter("xsd_to_type", filters::xsd_to_type);

        Ok(Self {
            tera,
            loaded_templates: Vec::new(),
        })
    }

    /// Load a custom template from a file
    pub fn load_template_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(SammError::ParseError(format!(
                "Template file not found: {}",
                path.display()
            )));
        }

        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| SammError::ParseError("Invalid template filename".to_string()))?;

        self.tera
            .add_template_file(path, Some(name))
            .map_err(|e| SammError::ParseError(format!("Failed to load template: {}", e)))?;

        self.loaded_templates.push(path.to_path_buf());
        tracing::debug!("Loaded custom template: {}", name);

        Ok(())
    }

    /// Load all templates from a directory
    pub fn load_template_dir<P: AsRef<Path>>(&mut self, dir: P) -> Result<()> {
        let dir = dir.as_ref();

        if !dir.is_dir() {
            return Err(SammError::ParseError(format!(
                "Template directory not found: {}",
                dir.display()
            )));
        }

        let pattern = dir.join("**/*.tera");
        let pattern_str = pattern
            .to_str()
            .ok_or_else(|| SammError::ParseError("Invalid template directory path".to_string()))?;

        self.tera
            .add_template_files(vec![(pattern_str, None::<String>)])
            .map_err(|e| {
                SammError::ParseError(format!("Failed to load template directory: {}", e))
            })?;

        tracing::info!("Loaded templates from directory: {}", dir.display());

        Ok(())
    }

    /// Render a template with the given context
    pub fn render(&self, template_name: &str, context: &TemplateContext) -> Result<String> {
        let tera_context = context.to_tera_context();

        self.tera
            .render(template_name, &tera_context)
            .map_err(|e| SammError::ParseError(format!("Template rendering failed: {}", e)))
    }

    /// Get list of available templates
    pub fn list_templates(&self) -> Vec<String> {
        self.tera.get_template_names().map(String::from).collect()
    }

    /// Check if a template exists
    pub fn has_template(&self, name: &str) -> bool {
        self.tera.get_template_names().any(|t| t == name)
    }
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create default template engine")
    }
}

/// Template context for rendering
#[derive(Debug, Clone)]
pub struct TemplateContext {
    /// Context data as JSON values
    data: HashMap<String, Value>,
}

impl TemplateContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Build context from an Aspect Model
    pub fn from_aspect(aspect: &Aspect) -> Self {
        let mut context = Self::new();

        context.insert("aspect_name", aspect.name());
        context.insert("namespace", aspect.urn());

        // Add properties
        let properties: Vec<Value> = aspect
            .properties()
            .iter()
            .map(|p| {
                json!({
                    "name": p.name(),
                    "urn": p.urn(),
                    "optional": p.optional,
                    "data_type": "xsd:string", // Default, should be extracted from characteristic
                })
            })
            .collect();
        context.insert("properties", properties);

        // Add operations
        let operations: Vec<Value> = aspect
            .operations()
            .iter()
            .map(|op| {
                json!({
                    "name": op.name(),
                    "urn": op.urn(),
                })
            })
            .collect();
        context.insert("operations", operations);

        context
    }

    /// Insert a value into the context
    pub fn insert<T: Into<Value>>(&mut self, key: impl Into<String>, value: T) {
        self.data.insert(key.into(), value.into());
    }

    /// Get a value from the context
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.data.get(key)
    }

    /// Convert to Tera context
    fn to_tera_context(&self) -> Context {
        let mut context = Context::new();
        for (key, value) in &self.data {
            context.insert(key, value);
        }
        context
    }
}

impl Default for TemplateContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Custom filters for template rendering
mod filters {
    use std::collections::HashMap;
    use tera::{Result as TeraResult, Value};

    /// Convert string to snake_case
    pub fn snake_case(value: &Value, _args: &HashMap<String, Value>) -> TeraResult<Value> {
        if let Some(s) = value.as_str() {
            let result = s
                .chars()
                .enumerate()
                .map(|(i, c)| {
                    if c.is_uppercase() && i > 0 {
                        format!("_{}", c.to_lowercase())
                    } else {
                        c.to_lowercase().to_string()
                    }
                })
                .collect::<String>();
            Ok(Value::String(result))
        } else {
            Ok(value.clone())
        }
    }

    /// Convert string to camelCase
    pub fn camel_case(value: &Value, _args: &HashMap<String, Value>) -> TeraResult<Value> {
        if let Some(s) = value.as_str() {
            let mut result = String::new();
            let mut capitalize_next = false;

            for (i, c) in s.chars().enumerate() {
                if c == '_' || c == '-' || c == ' ' {
                    capitalize_next = true;
                } else if capitalize_next {
                    result.push(c.to_uppercase().next().unwrap());
                    capitalize_next = false;
                } else if i == 0 {
                    result.push(c.to_lowercase().next().unwrap());
                } else {
                    result.push(c);
                }
            }

            Ok(Value::String(result))
        } else {
            Ok(value.clone())
        }
    }

    /// Convert string to PascalCase
    pub fn pascal_case(value: &Value, _args: &HashMap<String, Value>) -> TeraResult<Value> {
        if let Some(s) = value.as_str() {
            let mut result = String::new();
            let mut capitalize_next = true;

            for c in s.chars() {
                if c == '_' || c == '-' || c == ' ' {
                    capitalize_next = true;
                } else if capitalize_next {
                    result.push(c.to_uppercase().next().unwrap());
                    capitalize_next = false;
                } else {
                    result.push(c);
                }
            }

            Ok(Value::String(result))
        } else {
            Ok(value.clone())
        }
    }

    /// Convert string to kebab-case
    pub fn kebab_case(value: &Value, _args: &HashMap<String, Value>) -> TeraResult<Value> {
        if let Some(s) = value.as_str() {
            let result = s
                .chars()
                .enumerate()
                .map(|(i, c)| {
                    if c.is_uppercase() && i > 0 {
                        format!("-{}", c.to_lowercase())
                    } else if c == '_' {
                        "-".to_string()
                    } else {
                        c.to_lowercase().to_string()
                    }
                })
                .collect::<String>();
            Ok(Value::String(result))
        } else {
            Ok(value.clone())
        }
    }

    /// Convert string to UPPER_CASE
    pub fn upper_case(value: &Value, _args: &HashMap<String, Value>) -> TeraResult<Value> {
        if let Some(s) = value.as_str() {
            Ok(Value::String(s.to_uppercase()))
        } else {
            Ok(value.clone())
        }
    }

    /// Map XSD type to target language type
    pub fn xsd_to_type(value: &Value, args: &HashMap<String, Value>) -> TeraResult<Value> {
        if let Some(xsd_type) = value.as_str() {
            let target = args
                .get("target")
                .and_then(|v| v.as_str())
                .unwrap_or("rust");

            let mapped_type = match (target, xsd_type) {
                ("rust", "xsd:string") => "String",
                ("rust", "xsd:int") | ("rust", "xsd:integer") => "i64",
                ("rust", "xsd:float") => "f32",
                ("rust", "xsd:double") => "f64",
                ("rust", "xsd:boolean") => "bool",
                ("rust", "xsd:date") => "chrono::NaiveDate",
                ("rust", "xsd:dateTime") => "chrono::DateTime<chrono::Utc>",

                ("python", "xsd:string") => "str",
                ("python", "xsd:int") | ("python", "xsd:integer") => "int",
                ("python", "xsd:float") | ("python", "xsd:double") => "float",
                ("python", "xsd:boolean") => "bool",
                ("python", "xsd:date") | ("python", "xsd:dateTime") => "datetime.datetime",

                ("typescript", "xsd:string") => "string",
                ("typescript", "xsd:int") | ("typescript", "xsd:integer") => "number",
                ("typescript", "xsd:float") | ("typescript", "xsd:double") => "number",
                ("typescript", "xsd:boolean") => "boolean",
                ("typescript", "xsd:date") | ("typescript", "xsd:dateTime") => "Date",

                ("java", "xsd:string") => "String",
                ("java", "xsd:int") | ("java", "xsd:integer") => "Long",
                ("java", "xsd:float") => "Float",
                ("java", "xsd:double") => "Double",
                ("java", "xsd:boolean") => "Boolean",
                ("java", "xsd:date") => "java.time.LocalDate",
                ("java", "xsd:dateTime") => "java.time.ZonedDateTime",

                ("graphql", "xsd:string") => "String",
                ("graphql", "xsd:int") | ("graphql", "xsd:integer") => "Int",
                ("graphql", "xsd:float") | ("graphql", "xsd:double") => "Float",
                ("graphql", "xsd:boolean") => "Boolean",
                ("graphql", "xsd:date") | ("graphql", "xsd:dateTime") => "DateTime",

                _ => "any", // Fallback
            };

            Ok(Value::String(mapped_type.to_string()))
        } else {
            Ok(value.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_engine_creation() {
        let engine = TemplateEngine::new().unwrap();
        assert!(engine.has_template("rust.tera"));
        assert!(engine.has_template("python.tera"));
        assert!(engine.has_template("typescript.tera"));
    }

    #[test]
    fn test_list_templates() {
        let engine = TemplateEngine::new().unwrap();
        let templates = engine.list_templates();
        assert!(templates.contains(&"rust.tera".to_string()));
        assert!(templates.len() >= 5);
    }

    #[test]
    fn test_context_creation() {
        let mut context = TemplateContext::new();
        context.insert("name", "Test");
        context.insert("count", 42);

        assert_eq!(context.get("name").unwrap().as_str(), Some("Test"));
        assert_eq!(context.get("count").unwrap().as_i64(), Some(42));
    }

    #[test]
    fn test_snake_case_filter() {
        use std::collections::HashMap;

        let value = Value::String("MyPropertyName".to_string());
        let result = filters::snake_case(&value, &HashMap::new()).unwrap();
        assert_eq!(result.as_str().unwrap(), "my_property_name");
    }

    #[test]
    fn test_camel_case_filter() {
        use std::collections::HashMap;

        let value = Value::String("my_property_name".to_string());
        let result = filters::camel_case(&value, &HashMap::new()).unwrap();
        assert_eq!(result.as_str().unwrap(), "myPropertyName");
    }

    #[test]
    fn test_pascal_case_filter() {
        use std::collections::HashMap;

        let value = Value::String("my_property_name".to_string());
        let result = filters::pascal_case(&value, &HashMap::new()).unwrap();
        assert_eq!(result.as_str().unwrap(), "MyPropertyName");
    }
}
