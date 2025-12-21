//! SPARQL Query Template System
//!
//! Provides reusable query templates for common SPARQL patterns, enabling
//! rapid query development and consistent query structure.
//!
//! ## Features
//!
//! - **Predefined templates**: Common patterns (CRUD, search, aggregation)
//! - **Parameter substitution**: Type-safe parameter binding
//! - **Template composition**: Combine templates for complex queries
//! - **Custom templates**: Define and register custom templates
//! - **Validation**: Automatic parameter validation
//! - **Best practices**: Templates follow SPARQL optimization guidelines
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxirs_arq::query_templates::{TemplateRegistry, TemplateParams};
//!
//! let registry = TemplateRegistry::with_defaults();
//!
//! let mut params = TemplateParams::new();
//! params.set("predicate", "foaf:name");
//! params.set("value", "\"Alice\"");
//!
//! let query = registry.render("find_by_property", &params)?;
//! println!("{}", query);
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Parameter values for template rendering
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemplateParams {
    params: HashMap<String, String>,
}

impl TemplateParams {
    /// Create new empty parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a parameter value
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.params.insert(key.into(), value.into());
        self
    }

    /// Get a parameter value
    pub fn get(&self, key: &str) -> Option<&String> {
        self.params.get(key)
    }

    /// Check if parameter exists
    pub fn has(&self, key: &str) -> bool {
        self.params.contains_key(key)
    }

    /// Get all parameter keys
    pub fn keys(&self) -> Vec<&String> {
        self.params.keys().collect()
    }

    /// Builder-style parameter setting
    pub fn with(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.set(key, value);
        self
    }
}

/// SPARQL query template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTemplate {
    /// Template name
    pub name: String,

    /// Template description
    pub description: String,

    /// Template body with placeholders like {{variable}}
    pub template: String,

    /// Required parameter names
    pub required_params: Vec<String>,

    /// Optional parameter names with defaults
    pub optional_params: HashMap<String, String>,

    /// Template category
    pub category: TemplateCategory,

    /// Example usage
    pub example: Option<String>,
}

/// Template category for organization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemplateCategory {
    /// Data retrieval queries
    Retrieval,

    /// Data modification queries
    Modification,

    /// Aggregation and analytics
    Aggregation,

    /// Graph pattern matching
    PatternMatching,

    /// Full-text search
    Search,

    /// Administrative queries
    Admin,

    /// Custom user templates
    Custom,
}

impl QueryTemplate {
    /// Create a new template
    pub fn new(name: impl Into<String>, template: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            template: template.into(),
            required_params: Vec::new(),
            optional_params: HashMap::new(),
            category: TemplateCategory::Custom,
            example: None,
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add required parameter
    pub fn with_required(mut self, param: impl Into<String>) -> Self {
        self.required_params.push(param.into());
        self
    }

    /// Add optional parameter with default
    pub fn with_optional(mut self, param: impl Into<String>, default: impl Into<String>) -> Self {
        self.optional_params.insert(param.into(), default.into());
        self
    }

    /// Set category
    pub fn with_category(mut self, category: TemplateCategory) -> Self {
        self.category = category;
        self
    }

    /// Set example
    pub fn with_example(mut self, example: impl Into<String>) -> Self {
        self.example = Some(example.into());
        self
    }

    /// Render template with parameters
    pub fn render(&self, params: &TemplateParams) -> Result<String> {
        // Validate required parameters
        for required in &self.required_params {
            if !params.has(required) {
                return Err(anyhow!(
                    "Missing required parameter '{}' for template '{}'",
                    required,
                    self.name
                ));
            }
        }

        let mut result = self.template.clone();

        // Substitute required parameters
        for (key, value) in &params.params {
            let placeholder = format!("{{{{{}}}}}", key);
            result = result.replace(&placeholder, value);
        }

        // Substitute optional parameters with defaults
        for (key, default) in &self.optional_params {
            let placeholder = format!("{{{{{}}}}}", key);
            if result.contains(&placeholder) {
                let value = params.get(key).unwrap_or(default);
                result = result.replace(&placeholder, value);
            }
        }

        // Check for unsubstituted placeholders
        if result.contains("{{") {
            return Err(anyhow!(
                "Template '{}' contains unsubstituted placeholders",
                self.name
            ));
        }

        Ok(result)
    }
}

/// Registry of query templates
pub struct TemplateRegistry {
    templates: HashMap<String, QueryTemplate>,
}

impl TemplateRegistry {
    /// Create empty registry
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    /// Create registry with default templates
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register_default_templates();
        registry
    }

    /// Register a template
    pub fn register(&mut self, template: QueryTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    /// Get a template by name
    pub fn get(&self, name: &str) -> Option<&QueryTemplate> {
        self.templates.get(name)
    }

    /// Render a template with parameters
    pub fn render(&self, name: &str, params: &TemplateParams) -> Result<String> {
        let template = self
            .get(name)
            .ok_or_else(|| anyhow!("Template '{}' not found", name))?;

        template.render(params)
    }

    /// List all template names
    pub fn list_templates(&self) -> Vec<&String> {
        self.templates.keys().collect()
    }

    /// List templates by category
    pub fn list_by_category(&self, category: TemplateCategory) -> Vec<&QueryTemplate> {
        self.templates
            .values()
            .filter(|t| t.category == category)
            .collect()
    }

    /// Register default templates
    fn register_default_templates(&mut self) {
        // Find by property value
        self.register(
            QueryTemplate::new(
                "find_by_property",
                "SELECT ?subject WHERE {\n  ?subject {{predicate}} {{value}} .\n} LIMIT {{limit}}",
            )
            .with_description("Find resources by property value")
            .with_required("predicate")
            .with_required("value")
            .with_optional("limit", "100")
            .with_category(TemplateCategory::Retrieval)
            .with_example("Find all people named Alice"),
        );

        // Get all properties of a resource
        self.register(
            QueryTemplate::new(
                "get_all_properties",
                "SELECT ?property ?value WHERE {\n  {{subject}} ?property ?value .\n} LIMIT {{limit}}"
            )
            .with_description("Get all properties of a specific resource")
            .with_required("subject")
            .with_optional("limit", "100")
            .with_category(TemplateCategory::Retrieval)
        );

        // Count by type
        self.register(
            QueryTemplate::new(
                "count_by_type",
                "SELECT (COUNT(?subject) AS ?count) WHERE {\n  ?subject a {{type}} .\n}",
            )
            .with_description("Count resources of a specific type")
            .with_required("type")
            .with_category(TemplateCategory::Aggregation),
        );

        // Search by label
        self.register(
            QueryTemplate::new(
                "search_by_label",
                "SELECT ?subject ?label WHERE {\n  ?subject rdfs:label ?label .\n  FILTER(CONTAINS(LCASE(?label), LCASE({{search_term}})))\n} LIMIT {{limit}}"
            )
            .with_description("Search resources by label (case-insensitive)")
            .with_required("search_term")
            .with_optional("limit", "50")
            .with_category(TemplateCategory::Search)
        );

        // Insert triple
        self.register(
            QueryTemplate::new(
                "insert_triple",
                "INSERT DATA {\n  {{subject}} {{predicate}} {{object}} .\n}",
            )
            .with_description("Insert a single triple")
            .with_required("subject")
            .with_required("predicate")
            .with_required("object")
            .with_category(TemplateCategory::Modification),
        );

        // Delete by pattern
        self.register(
            QueryTemplate::new(
                "delete_by_pattern",
                "DELETE WHERE {\n  ?subject {{predicate}} {{value}} .\n}",
            )
            .with_description("Delete triples matching a pattern")
            .with_required("predicate")
            .with_required("value")
            .with_category(TemplateCategory::Modification),
        );

        // Get distinct values
        self.register(
            QueryTemplate::new(
                "distinct_values",
                "SELECT DISTINCT ?value WHERE {\n  ?subject {{predicate}} ?value .\n} ORDER BY ?value LIMIT {{limit}}"
            )
            .with_description("Get distinct values for a property")
            .with_required("predicate")
            .with_optional("limit", "100")
            .with_category(TemplateCategory::Retrieval)
        );

        // Aggregate by group
        self.register(
            QueryTemplate::new(
                "group_count",
                "SELECT ?{{group_var}} (COUNT(*) AS ?count) WHERE {\n  {{pattern}}\n} GROUP BY ?{{group_var}} ORDER BY DESC(?count) LIMIT {{limit}}"
            )
            .with_description("Count occurrences grouped by a variable")
            .with_required("group_var")
            .with_required("pattern")
            .with_optional("limit", "50")
            .with_category(TemplateCategory::Aggregation)
        );

        // Property path query
        self.register(
            QueryTemplate::new(
                "property_path",
                "SELECT ?target WHERE {\n  {{source}} {{path}} ?target .\n} LIMIT {{limit}}",
            )
            .with_description("Query using property paths (e.g., rdfs:subClassOf*)")
            .with_required("source")
            .with_required("path")
            .with_optional("limit", "100")
            .with_category(TemplateCategory::PatternMatching),
        );

        // Optional pattern
        self.register(
            QueryTemplate::new(
                "with_optional",
                "SELECT ?subject ?required ?optional WHERE {\n  ?subject {{required_predicate}} ?required .\n  OPTIONAL { ?subject {{optional_predicate}} ?optional . }\n} LIMIT {{limit}}"
            )
            .with_description("Query with required and optional patterns")
            .with_required("required_predicate")
            .with_required("optional_predicate")
            .with_optional("limit", "100")
            .with_category(TemplateCategory::PatternMatching)
        );
    }
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_params() {
        let mut params = TemplateParams::new();
        params.set("key1", "value1");
        params.set("key2", "value2");

        assert_eq!(params.get("key1"), Some(&"value1".to_string()));
        assert_eq!(params.get("key2"), Some(&"value2".to_string()));
        assert!(params.has("key1"));
        assert!(!params.has("key3"));
    }

    #[test]
    fn test_template_params_builder() {
        let params = TemplateParams::new()
            .with("name", "Alice")
            .with("age", "30");

        assert_eq!(params.get("name"), Some(&"Alice".to_string()));
        assert_eq!(params.get("age"), Some(&"30".to_string()));
    }

    #[test]
    fn test_simple_template_render() {
        let template = QueryTemplate::new("test", "SELECT * WHERE { ?s {{predicate}} {{object}} }")
            .with_required("predicate")
            .with_required("object");

        let params = TemplateParams::new()
            .with("predicate", "foaf:name")
            .with("object", "\"Alice\"");

        let result = template.render(&params).unwrap();
        assert!(result.contains("foaf:name"));
        assert!(result.contains("\"Alice\""));
    }

    #[test]
    fn test_missing_required_param() {
        let template = QueryTemplate::new("test", "SELECT * WHERE { ?s {{predicate}} ?o }")
            .with_required("predicate");

        let params = TemplateParams::new();
        let result = template.render(&params);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Missing required parameter"));
    }

    #[test]
    fn test_optional_params() {
        let template = QueryTemplate::new("test", "SELECT * WHERE { ?s ?p ?o } LIMIT {{limit}}")
            .with_optional("limit", "100");

        // Without providing limit
        let params1 = TemplateParams::new();
        let result1 = template.render(&params1).unwrap();
        assert!(result1.contains("LIMIT 100"));

        // With custom limit
        let params2 = TemplateParams::new().with("limit", "50");
        let result2 = template.render(&params2).unwrap();
        assert!(result2.contains("LIMIT 50"));
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = TemplateRegistry::new();
        let template = QueryTemplate::new("test", "SELECT * WHERE { ?s ?p ?o }");

        registry.register(template);

        assert!(registry.get("test").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_render() {
        let mut registry = TemplateRegistry::new();
        registry.register(
            QueryTemplate::new("test", "SELECT * WHERE { ?s {{pred}} ?o }").with_required("pred"),
        );

        let params = TemplateParams::new().with("pred", "foaf:name");
        let result = registry.render("test", &params).unwrap();

        assert!(result.contains("foaf:name"));
    }

    #[test]
    fn test_default_templates() {
        let registry = TemplateRegistry::with_defaults();

        assert!(registry.get("find_by_property").is_some());
        assert!(registry.get("get_all_properties").is_some());
        assert!(registry.get("count_by_type").is_some());
        assert!(registry.get("search_by_label").is_some());
    }

    #[test]
    fn test_find_by_property_template() {
        let registry = TemplateRegistry::with_defaults();

        let params = TemplateParams::new()
            .with("predicate", "foaf:name")
            .with("value", "\"Alice\"");

        let query = registry.render("find_by_property", &params).unwrap();

        assert!(query.contains("foaf:name"));
        assert!(query.contains("\"Alice\""));
        assert!(query.contains("LIMIT 100")); // default limit
    }

    #[test]
    fn test_count_by_type_template() {
        let registry = TemplateRegistry::with_defaults();

        let params = TemplateParams::new().with("type", "foaf:Person");

        let query = registry.render("count_by_type", &params).unwrap();

        assert!(query.contains("COUNT"));
        assert!(query.contains("foaf:Person"));
    }

    #[test]
    fn test_insert_triple_template() {
        let registry = TemplateRegistry::with_defaults();

        let params = TemplateParams::new()
            .with("subject", "<http://example.org/alice>")
            .with("predicate", "foaf:name")
            .with("object", "\"Alice\"");

        let query = registry.render("insert_triple", &params).unwrap();

        assert!(query.contains("INSERT DATA"));
        assert!(query.contains("<http://example.org/alice>"));
        assert!(query.contains("foaf:name"));
        assert!(query.contains("\"Alice\""));
    }

    #[test]
    fn test_list_templates() {
        let registry = TemplateRegistry::with_defaults();
        let templates = registry.list_templates();

        assert!(!templates.is_empty());
        assert!(templates.contains(&&"find_by_property".to_string()));
    }

    #[test]
    fn test_list_by_category() {
        let registry = TemplateRegistry::with_defaults();
        let retrieval = registry.list_by_category(TemplateCategory::Retrieval);
        let aggregation = registry.list_by_category(TemplateCategory::Aggregation);

        assert!(!retrieval.is_empty());
        assert!(!aggregation.is_empty());

        // Verify categories
        for template in retrieval {
            assert_eq!(template.category, TemplateCategory::Retrieval);
        }
    }

    #[test]
    fn test_search_by_label_template() {
        let registry = TemplateRegistry::with_defaults();

        let params = TemplateParams::new().with("search_term", "\"person\"");

        let query = registry.render("search_by_label", &params).unwrap();

        assert!(query.contains("rdfs:label"));
        assert!(query.contains("FILTER"));
        assert!(query.contains("CONTAINS"));
    }

    #[test]
    fn test_template_with_custom_limit() {
        let registry = TemplateRegistry::with_defaults();

        let params = TemplateParams::new()
            .with("predicate", "foaf:name")
            .with("value", "\"Alice\"")
            .with("limit", "500");

        let query = registry.render("find_by_property", &params).unwrap();

        assert!(query.contains("LIMIT 500"));
    }
}
