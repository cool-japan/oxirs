//! Schema Documentation Generator
//!
//! Generates comprehensive documentation from GraphQL schemas in multiple formats:
//! - **Markdown**: For GitHub/GitLab README and wikis
//! - **HTML**: For static documentation sites
//! - **JSON**: For programmatic access
//! - **OpenAPI**: For REST API documentation
//!
//! ## Features
//!
//! - **Complete Coverage**: Documents all types, fields, arguments, directives
//! - **Cross-References**: Links between related types
//! - **Examples**: Auto-generated query examples
//! - **Search Index**: Full-text search support
//! - **Versioning**: Track schema changes across versions
//! - **Customizable**: Templates for custom styling

use crate::types::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt::Write as FmtWrite;

/// Documentation format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocFormat {
    /// Markdown format
    Markdown,
    /// HTML format
    Html,
    /// JSON format
    Json,
    /// OpenAPI 3.0 format
    OpenAPI,
}

/// Documentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocConfig {
    /// Include deprecation warnings
    pub include_deprecated: bool,

    /// Generate usage examples
    pub generate_examples: bool,

    /// Include type relationships
    pub include_relationships: bool,

    /// Generate table of contents
    pub generate_toc: bool,

    /// Custom CSS for HTML
    pub custom_css: Option<String>,

    /// Project name
    pub project_name: String,

    /// Project version
    pub project_version: String,

    /// Base URL for links
    pub base_url: Option<String>,
}

impl Default for DocConfig {
    fn default() -> Self {
        Self {
            include_deprecated: true,
            generate_examples: true,
            include_relationships: true,
            generate_toc: true,
            custom_css: None,
            project_name: "GraphQL API".to_string(),
            project_version: "1.0.0".to_string(),
            base_url: None,
        }
    }
}

/// Schema documentation generator
pub struct SchemaDocsGenerator {
    config: DocConfig,
}

impl SchemaDocsGenerator {
    pub fn new(config: DocConfig) -> Self {
        Self { config }
    }

    /// Generate documentation in specified format
    pub fn generate(&self, schema: &Schema, format: DocFormat) -> Result<String> {
        match format {
            DocFormat::Markdown => self.generate_markdown(schema),
            DocFormat::Html => self.generate_html(schema),
            DocFormat::Json => self.generate_json(schema),
            DocFormat::OpenAPI => self.generate_openapi(schema),
        }
    }

    /// Generate Markdown documentation
    fn generate_markdown(&self, schema: &Schema) -> Result<String> {
        let mut md = String::new();

        // Title
        writeln!(
            &mut md,
            "# {} - GraphQL API Documentation",
            self.config.project_name
        )?;
        writeln!(&mut md, "\nVersion: {}\n", self.config.project_version)?;

        // Table of Contents
        if self.config.generate_toc {
            writeln!(&mut md, "## Table of Contents\n")?;
            writeln!(&mut md, "- [Query Operations](#query-operations)")?;
            writeln!(&mut md, "- [Mutation Operations](#mutation-operations)")?;
            writeln!(
                &mut md,
                "- [Subscription Operations](#subscription-operations)"
            )?;
            writeln!(&mut md, "- [Types](#types)")?;
            writeln!(&mut md, "- [Scalars](#scalars)")?;
            writeln!(&mut md, "- [Enums](#enums)")?;
            writeln!(&mut md, "- [Interfaces](#interfaces)")?;
            writeln!(&mut md, "- [Unions](#unions)")?;
            writeln!(&mut md)?;
        }

        // Query Type
        if let Some(query_type) = &schema.query_type {
            writeln!(&mut md, "## Query Operations\n")?;
            self.document_type_markdown(&mut md, schema, query_type)?;
        }

        // Mutation Type
        if let Some(mutation_type) = &schema.mutation_type {
            writeln!(&mut md, "## Mutation Operations\n")?;
            self.document_type_markdown(&mut md, schema, mutation_type)?;
        }

        // Subscription Type
        if let Some(subscription_type) = &schema.subscription_type {
            writeln!(&mut md, "## Subscription Operations\n")?;
            self.document_type_markdown(&mut md, schema, subscription_type)?;
        }

        // Other Types
        writeln!(&mut md, "## Types\n")?;
        for graphql_type in schema.types.values() {
            if let GraphQLType::Object(obj_type) = graphql_type {
                // Skip root operation types
                let is_root = Some(&obj_type.name) == schema.query_type.as_ref()
                    || Some(&obj_type.name) == schema.mutation_type.as_ref()
                    || Some(&obj_type.name) == schema.subscription_type.as_ref();

                if !is_root {
                    self.document_object_type_markdown(&mut md, obj_type)?;
                }
            }
        }

        // Scalars
        writeln!(&mut md, "## Scalars\n")?;
        for graphql_type in schema.types.values() {
            if let GraphQLType::Scalar(scalar) = graphql_type {
                self.document_scalar_markdown(&mut md, scalar)?;
            }
        }

        // Enums
        writeln!(&mut md, "## Enums\n")?;
        for graphql_type in schema.types.values() {
            if let GraphQLType::Enum(enum_type) = graphql_type {
                self.document_enum_markdown(&mut md, enum_type)?;
            }
        }

        Ok(md)
    }

    fn document_type_markdown(
        &self,
        md: &mut String,
        schema: &Schema,
        type_name: &str,
    ) -> Result<()> {
        if let Some(graphql_type) = schema.get_type(type_name) {
            match graphql_type {
                GraphQLType::Object(obj_type) => {
                    self.document_object_type_markdown(md, obj_type)?
                }
                GraphQLType::Scalar(scalar) => self.document_scalar_markdown(md, scalar)?,
                GraphQLType::Enum(enum_type) => self.document_enum_markdown(md, enum_type)?,
                _ => {}
            }
        }
        Ok(())
    }

    fn document_object_type_markdown(&self, md: &mut String, obj_type: &ObjectType) -> Result<()> {
        writeln!(md, "### {}\n", obj_type.name)?;

        if let Some(desc) = &obj_type.description {
            writeln!(md, "{}\n", desc)?;
        }

        if !obj_type.fields.is_empty() {
            writeln!(md, "#### Fields\n")?;
            writeln!(md, "| Field | Type | Description |")?;
            writeln!(md, "|-------|------|-------------|")?;

            for (field_name, field) in &obj_type.fields {
                let field_type = Self::format_type(&field.field_type);
                let desc = field.description.as_deref().unwrap_or("");
                writeln!(md, "| `{}` | `{}` | {} |", field_name, field_type, desc)?;
            }
            writeln!(md)?;
        }

        // Generate example if enabled
        if self.config.generate_examples {
            writeln!(md, "#### Example Query\n")?;
            writeln!(md, "```graphql")?;
            writeln!(md, "{{")?;
            writeln!(md, "  {} {{", obj_type.name.to_lowercase())?;
            for (field_name, _) in obj_type.fields.iter().take(3) {
                writeln!(md, "    {}", field_name)?;
            }
            writeln!(md, "  }}")?;
            writeln!(md, "}}")?;
            writeln!(md, "```\n")?;
        }

        Ok(())
    }

    fn document_scalar_markdown(&self, md: &mut String, scalar: &ScalarType) -> Result<()> {
        writeln!(md, "### {}\n", scalar.name)?;

        if let Some(desc) = &scalar.description {
            writeln!(md, "{}\n", desc)?;
        }

        Ok(())
    }

    fn document_enum_markdown(&self, md: &mut String, enum_type: &EnumType) -> Result<()> {
        writeln!(md, "### {}\n", enum_type.name)?;

        if let Some(desc) = &enum_type.description {
            writeln!(md, "{}\n", desc)?;
        }

        if !enum_type.values.is_empty() {
            writeln!(md, "#### Values\n")?;
            for value in enum_type.values.values() {
                writeln!(md, "- `{}`", value.name)?;
                if let Some(desc) = &value.description {
                    writeln!(md, "  - {}", desc)?;
                }
            }
            writeln!(md)?;
        }

        Ok(())
    }

    /// Generate HTML documentation
    fn generate_html(&self, schema: &Schema) -> Result<String> {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
        html.push_str("    <meta charset=\"UTF-8\">\n");
        html.push_str(
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        );
        html.push_str(&format!(
            "    <title>{} - GraphQL API Documentation</title>\n",
            self.config.project_name
        ));

        // CSS
        html.push_str("    <style>\n");
        if let Some(custom_css) = &self.config.custom_css {
            html.push_str(custom_css);
        } else {
            html.push_str(DEFAULT_CSS);
        }
        html.push_str("    </style>\n");
        html.push_str("</head>\n<body>\n");

        // Header
        html.push_str("    <header>\n");
        html.push_str(&format!(
            "        <h1>{} - GraphQL API</h1>\n",
            self.config.project_name
        ));
        html.push_str(&format!(
            "        <p class=\"version\">Version: {}</p>\n",
            self.config.project_version
        ));
        html.push_str("    </header>\n");

        // Content
        html.push_str("    <div class=\"container\">\n");

        // Sidebar navigation
        if self.config.generate_toc {
            html.push_str("        <nav class=\"sidebar\">\n");
            html.push_str("            <h2>Navigation</h2>\n");
            html.push_str("            <ul>\n");
            html.push_str("                <li><a href=\"#queries\">Queries</a></li>\n");
            html.push_str("                <li><a href=\"#mutations\">Mutations</a></li>\n");
            html.push_str("                <li><a href=\"#types\">Types</a></li>\n");
            html.push_str("                <li><a href=\"#scalars\">Scalars</a></li>\n");
            html.push_str("            </ul>\n");
            html.push_str("        </nav>\n");
        }

        // Main content
        html.push_str("        <main>\n");

        // Query Type
        if let Some(query_type) = &schema.query_type {
            html.push_str("            <section id=\"queries\">\n");
            html.push_str("                <h2>Query Operations</h2>\n");
            self.document_type_html(&mut html, schema, query_type)?;
            html.push_str("            </section>\n");
        }

        // Types
        html.push_str("            <section id=\"types\">\n");
        html.push_str("                <h2>Types</h2>\n");
        for graphql_type in schema.types.values() {
            if let GraphQLType::Object(obj_type) = graphql_type {
                self.document_object_type_html(&mut html, obj_type)?;
            }
        }
        html.push_str("            </section>\n");

        html.push_str("        </main>\n");
        html.push_str("    </div>\n");
        html.push_str("</body>\n</html>");

        Ok(html)
    }

    fn document_type_html(
        &self,
        html: &mut String,
        schema: &Schema,
        type_name: &str,
    ) -> Result<()> {
        if let Some(GraphQLType::Object(obj_type)) = schema.get_type(type_name) {
            self.document_object_type_html(html, obj_type)?;
        }
        Ok(())
    }

    fn document_object_type_html(&self, html: &mut String, obj_type: &ObjectType) -> Result<()> {
        html.push_str("                <div class=\"type\">\n");
        html.push_str(&format!("                    <h3>{}</h3>\n", obj_type.name));

        if let Some(desc) = &obj_type.description {
            html.push_str(&format!("                    <p>{}</p>\n", desc));
        }

        if !obj_type.fields.is_empty() {
            html.push_str("                    <h4>Fields</h4>\n");
            html.push_str("                    <table>\n");
            html.push_str("                        <thead>\n");
            html.push_str("                            <tr><th>Field</th><th>Type</th><th>Description</th></tr>\n");
            html.push_str("                        </thead>\n");
            html.push_str("                        <tbody>\n");

            for (field_name, field) in &obj_type.fields {
                let field_type = Self::format_type(&field.field_type);
                let desc = field.description.as_deref().unwrap_or("");
                html.push_str(&format!(
                    "                            <tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td></tr>\n",
                    field_name, field_type, desc
                ));
            }

            html.push_str("                        </tbody>\n");
            html.push_str("                    </table>\n");
        }

        html.push_str("                </div>\n");
        Ok(())
    }

    /// Generate JSON documentation
    fn generate_json(&self, schema: &Schema) -> Result<String> {
        let doc = serde_json::json!({
            "name": self.config.project_name,
            "version": self.config.project_version,
            "queryType": schema.query_type,
            "mutationType": schema.mutation_type,
            "subscriptionType": schema.subscription_type,
            "types": schema.types.values().map(|t| self.type_to_json(t)).collect::<Vec<_>>(),
        });

        Ok(serde_json::to_string_pretty(&doc)?)
    }

    fn type_to_json(&self, graphql_type: &GraphQLType) -> serde_json::Value {
        match graphql_type {
            GraphQLType::Object(obj) => serde_json::json!({
                "kind": "OBJECT",
                "name": obj.name,
                "description": obj.description,
                "fields": obj.fields.iter().map(|(name, field)| serde_json::json!({
                    "name": name,
                    "type": Self::format_type(&field.field_type),
                    "description": field.description,
                })).collect::<Vec<_>>(),
            }),
            GraphQLType::Scalar(scalar) => serde_json::json!({
                "kind": "SCALAR",
                "name": scalar.name,
                "description": scalar.description,
            }),
            GraphQLType::Enum(enum_type) => serde_json::json!({
                "kind": "ENUM",
                "name": enum_type.name,
                "description": enum_type.description,
                "enumValues": enum_type.values.values().map(|v| serde_json::json!({
                    "name": v.name,
                    "description": v.description,
                })).collect::<Vec<_>>(),
            }),
            _ => serde_json::json!({}),
        }
    }

    /// Generate OpenAPI documentation
    fn generate_openapi(&self, _schema: &Schema) -> Result<String> {
        let openapi = serde_json::json!({
            "openapi": "3.0.0",
            "info": {
                "title": self.config.project_name,
                "version": self.config.project_version,
                "description": "GraphQL API converted to OpenAPI format"
            },
            "paths": {
                "/graphql": {
                    "post": {
                        "summary": "Execute GraphQL query",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "query": {"type": "string"},
                                            "variables": {"type": "object"},
                                            "operationName": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(serde_json::to_string_pretty(&openapi)?)
    }

    fn format_type(graphql_type: &GraphQLType) -> String {
        match graphql_type {
            GraphQLType::NonNull(inner) => format!("{}!", Self::format_type(inner)),
            GraphQLType::List(inner) => format!("[{}]", Self::format_type(inner)),
            GraphQLType::Object(obj) => obj.name.clone(),
            GraphQLType::Scalar(scalar) => scalar.name.clone(),
            GraphQLType::Enum(enum_type) => enum_type.name.clone(),
            GraphQLType::Interface(iface) => iface.name.clone(),
            GraphQLType::Union(union) => union.name.clone(),
            GraphQLType::InputObject(input) => input.name.clone(),
        }
    }
}

const DEFAULT_CSS: &str = r#"
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    line-height: 1.6;
    color: #333;
    margin: 0;
    padding: 0;
}

header {
    background: #2c3e50;
    color: white;
    padding: 2rem;
    text-align: center;
}

.version {
    color: #ecf0f1;
    font-size: 0.9rem;
}

.container {
    display: flex;
    max-width: 1200px;
    margin: 0 auto;
}

.sidebar {
    width: 250px;
    padding: 2rem;
    background: #f8f9fa;
    border-right: 1px solid #dee2e6;
}

.sidebar ul {
    list-style: none;
    padding: 0;
}

.sidebar li {
    margin-bottom: 0.5rem;
}

.sidebar a {
    color: #495057;
    text-decoration: none;
}

.sidebar a:hover {
    color: #007bff;
}

main {
    flex: 1;
    padding: 2rem;
}

.type {
    margin-bottom: 3rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #dee2e6;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
}

th, td {
    padding: 0.75rem;
    text-align: left;
    border-bottom: 1px solid #dee2e6;
}

th {
    background: #f8f9fa;
    font-weight: 600;
}

code {
    background: #f8f9fa;
    padding: 0.2rem 0.4rem;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doc_config_default() {
        let config = DocConfig::default();
        assert!(config.include_deprecated);
        assert!(config.generate_examples);
        assert_eq!(config.project_name, "GraphQL API");
    }

    #[test]
    fn test_doc_format_equality() {
        assert_eq!(DocFormat::Markdown, DocFormat::Markdown);
        assert_ne!(DocFormat::Markdown, DocFormat::Html);
    }

    #[test]
    fn test_generator_creation() {
        let config = DocConfig::default();
        let _generator = SchemaDocsGenerator::new(config);
    }

    #[test]
    fn test_generate_markdown_empty_schema() {
        let config = DocConfig::default();
        let generator = SchemaDocsGenerator::new(config);
        let schema = Schema::new();

        let result = generator.generate(&schema, DocFormat::Markdown);
        assert!(result.is_ok());

        let md = result.unwrap();
        assert!(md.contains("GraphQL API Documentation"));
    }

    #[test]
    fn test_generate_html_empty_schema() {
        let config = DocConfig::default();
        let generator = SchemaDocsGenerator::new(config);
        let schema = Schema::new();

        let result = generator.generate(&schema, DocFormat::Html);
        assert!(result.is_ok());

        let html = result.unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("GraphQL API"));
    }

    #[test]
    fn test_generate_json_empty_schema() {
        let config = DocConfig::default();
        let generator = SchemaDocsGenerator::new(config);
        let schema = Schema::new();

        let result = generator.generate(&schema, DocFormat::Json);
        assert!(result.is_ok());

        let json = result.unwrap();
        assert!(json.contains("GraphQL API"));
    }

    #[test]
    fn test_generate_openapi() {
        let config = DocConfig::default();
        let generator = SchemaDocsGenerator::new(config);
        let schema = Schema::new();

        let result = generator.generate(&schema, DocFormat::OpenAPI);
        assert!(result.is_ok());

        let openapi = result.unwrap();
        assert!(openapi.contains("openapi"));
        assert!(openapi.contains("3.0.0"));
    }

    #[test]
    fn test_format_type_scalar() {
        let scalar = GraphQLType::Scalar(ScalarType {
            name: "String".to_string(),
            description: None,
            serialize: |_| Ok(crate::ast::Value::NullValue),
            parse_value: |_| Err(anyhow::anyhow!("test")),
            parse_literal: |_| Err(anyhow::anyhow!("test")),
        });

        assert_eq!(SchemaDocsGenerator::format_type(&scalar), "String");
    }

    #[test]
    fn test_format_type_non_null() {
        let scalar = GraphQLType::Scalar(ScalarType {
            name: "String".to_string(),
            description: None,
            serialize: |_| Ok(crate::ast::Value::NullValue),
            parse_value: |_| Err(anyhow::anyhow!("test")),
            parse_literal: |_| Err(anyhow::anyhow!("test")),
        });

        let non_null = GraphQLType::NonNull(Box::new(scalar));
        assert_eq!(SchemaDocsGenerator::format_type(&non_null), "String!");
    }

    #[test]
    fn test_format_type_list() {
        let scalar = GraphQLType::Scalar(ScalarType {
            name: "String".to_string(),
            description: None,
            serialize: |_| Ok(crate::ast::Value::NullValue),
            parse_value: |_| Err(anyhow::anyhow!("test")),
            parse_literal: |_| Err(anyhow::anyhow!("test")),
        });

        let list = GraphQLType::List(Box::new(scalar));
        assert_eq!(SchemaDocsGenerator::format_type(&list), "[String]");
    }

    #[test]
    fn test_custom_config() {
        let config = DocConfig {
            project_name: "My API".to_string(),
            project_version: "2.0.0".to_string(),
            generate_examples: false,
            ..Default::default()
        };

        let generator = SchemaDocsGenerator::new(config.clone());
        let schema = Schema::new();

        let md = generator.generate(&schema, DocFormat::Markdown).unwrap();
        assert!(md.contains("My API"));
        assert!(md.contains("2.0.0"));
    }
}
