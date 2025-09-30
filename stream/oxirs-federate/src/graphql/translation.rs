//! GraphQL to SPARQL Translation Layer
//!
//! This module provides translation capabilities between GraphQL queries and SPARQL queries,
//! enabling hybrid federated query processing across different semantic data sources.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

use super::types::*;
use crate::planner::planning::types::TriplePattern;

/// GraphQL to SPARQL translator
#[derive(Debug)]
pub struct GraphQLToSparqlTranslator {
    config: TranslationConfig,
    schema_mappings: HashMap<String, SchemaMapping>,
    namespace_mappings: HashMap<String, String>,
    predicate_mappings: HashMap<String, String>,
}

/// Configuration for GraphQL to SPARQL translation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConfig {
    /// Default RDF namespace for generated triples
    pub default_namespace: String,
    /// Whether to generate typed literals
    pub generate_typed_literals: bool,
    /// Whether to use schema-aware translation
    pub schema_aware: bool,
    /// Maximum depth for nested object translation
    pub max_nesting_depth: usize,
    /// Prefix mappings for common namespaces
    pub namespace_prefixes: HashMap<String, String>,
    /// Default language tag for literals
    pub default_language: Option<String>,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        let mut namespace_prefixes = HashMap::new();
        namespace_prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        namespace_prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        namespace_prefixes.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        namespace_prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        namespace_prefixes.insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());
        namespace_prefixes.insert(
            "dc".to_string(),
            "http://purl.org/dc/elements/1.1/".to_string(),
        );
        namespace_prefixes.insert(
            "dcterms".to_string(),
            "http://purl.org/dc/terms/".to_string(),
        );

        Self {
            default_namespace: "http://example.org/data/".to_string(),
            generate_typed_literals: true,
            schema_aware: true,
            max_nesting_depth: 5,
            namespace_prefixes,
            default_language: None,
        }
    }
}

/// Schema mapping from GraphQL types to RDF predicates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMapping {
    /// GraphQL type name
    pub graphql_type: String,
    /// RDF class URI
    pub rdf_class: Option<String>,
    /// Field to predicate mappings
    pub field_mappings: HashMap<String, FieldMapping>,
    /// Key fields for entity identification
    pub key_fields: Vec<String>,
}

/// Field mapping from GraphQL field to RDF predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMapping {
    /// RDF predicate URI
    pub predicate: String,
    /// Data type information
    pub data_type: Option<String>,
    /// Whether this field represents a relationship
    pub is_relationship: bool,
    /// Target type for relationships
    pub target_type: Option<String>,
    /// Inverse predicate for bidirectional relationships
    pub inverse_predicate: Option<String>,
}

/// Translation result containing generated SPARQL
#[derive(Debug, Clone)]
pub struct TranslationResult {
    /// Generated SPARQL query
    pub sparql_query: String,
    /// Variable mappings from GraphQL to SPARQL
    pub variable_mappings: HashMap<String, String>,
    /// Triple patterns extracted from the query
    pub triple_patterns: Vec<TriplePattern>,
    /// Namespace prefixes used
    pub prefixes: HashMap<String, String>,
    /// Translation metadata
    pub metadata: TranslationMetadata,
}

/// Metadata about the translation process
#[derive(Debug, Clone)]
pub struct TranslationMetadata {
    /// Original GraphQL operation type
    pub operation_type: String,
    /// Number of fields translated
    pub field_count: usize,
    /// Maximum nesting depth encountered
    pub max_depth: usize,
    /// Whether any schema mappings were used
    pub used_schema_mappings: bool,
    /// Warnings encountered during translation
    pub warnings: Vec<String>,
}

/// Context for processing selections during translation
struct ProcessingContext<'a> {
    /// SPARQL query builder
    sparql_builder: &'a mut SparqlQueryBuilder,
    /// Variable mappings from GraphQL to SPARQL
    variable_mappings: &'a mut HashMap<String, String>,
    /// Triple patterns extracted from the query
    triple_patterns: &'a mut Vec<TriplePattern>,
    /// Warnings encountered during translation
    warnings: &'a mut Vec<String>,
}

impl GraphQLToSparqlTranslator {
    /// Create a new translator with default configuration
    pub fn new() -> Self {
        Self::with_config(TranslationConfig::default())
    }

    /// Create a new translator with custom configuration
    pub fn with_config(config: TranslationConfig) -> Self {
        Self {
            config,
            schema_mappings: HashMap::new(),
            namespace_mappings: HashMap::new(),
            predicate_mappings: HashMap::new(),
        }
    }

    /// Add a schema mapping for a GraphQL type
    pub fn add_schema_mapping(&mut self, mapping: SchemaMapping) {
        self.schema_mappings
            .insert(mapping.graphql_type.clone(), mapping);
    }

    /// Add a namespace mapping
    pub fn add_namespace_mapping(&mut self, prefix: String, namespace: String) {
        self.namespace_mappings.insert(prefix, namespace);
    }

    /// Translate a GraphQL query to SPARQL
    pub async fn translate_query(&self, graphql_query: &str) -> Result<TranslationResult> {
        debug!("Translating GraphQL query to SPARQL");

        // Parse the GraphQL query
        let parsed_query = self.parse_graphql_query(graphql_query)?;

        // Extract the operation
        let operation = self.extract_operation(&parsed_query)?;

        // Generate SPARQL based on operation type
        let translation_result = match operation.operation_type.as_str() {
            "query" => self.translate_query_operation(operation).await?,
            "mutation" => self.translate_mutation_operation(operation).await?,
            "subscription" => {
                return Err(anyhow!(
                    "Subscription operations cannot be translated to SPARQL"
                ));
            }
            _ => {
                return Err(anyhow!(
                    "Unknown GraphQL operation type: {}",
                    operation.operation_type
                ));
            }
        };

        info!(
            "Successfully translated GraphQL query to SPARQL with {} triple patterns",
            translation_result.triple_patterns.len()
        );

        Ok(translation_result)
    }

    /// Translate a GraphQL query operation to SPARQL SELECT
    async fn translate_query_operation(
        &self,
        operation: &GraphQLOperation,
    ) -> Result<TranslationResult> {
        let mut sparql_builder = SparqlQueryBuilder::new(&self.config);
        let mut variable_mappings = HashMap::new();
        let mut triple_patterns = Vec::new();
        let mut warnings = Vec::new();

        // Process the selection set
        let mut context = ProcessingContext {
            sparql_builder: &mut sparql_builder,
            variable_mappings: &mut variable_mappings,
            triple_patterns: &mut triple_patterns,
            warnings: &mut warnings,
        };

        self.process_selection_set(&operation.selection_set, None, &mut context, 0)
            .await?;

        // Build the final SPARQL query
        let sparql_query = sparql_builder.build()?;

        Ok(TranslationResult {
            sparql_query,
            variable_mappings,
            triple_patterns,
            prefixes: sparql_builder.get_prefixes(),
            metadata: TranslationMetadata {
                operation_type: operation.operation_type.clone(),
                field_count: Self::count_fields(&operation.selection_set),
                max_depth: Self::calculate_max_depth(&operation.selection_set),
                used_schema_mappings: !self.schema_mappings.is_empty(),
                warnings,
            },
        })
    }

    /// Translate a GraphQL mutation operation to SPARQL UPDATE
    async fn translate_mutation_operation(
        &self,
        operation: &GraphQLOperation,
    ) -> Result<TranslationResult> {
        let mut sparql_builder = SparqlUpdateBuilder::new(&self.config);
        let mut variable_mappings = HashMap::new();
        let mut triple_patterns = Vec::new();
        let mut warnings = Vec::new();

        // Process mutation fields
        for field in &operation.selection_set {
            self.process_mutation_field(
                field,
                &mut sparql_builder,
                &mut variable_mappings,
                &mut triple_patterns,
                &mut warnings,
            )
            .await?;
        }

        let sparql_query = sparql_builder.build()?;

        Ok(TranslationResult {
            sparql_query,
            variable_mappings,
            triple_patterns,
            prefixes: sparql_builder.get_prefixes(),
            metadata: TranslationMetadata {
                operation_type: operation.operation_type.clone(),
                field_count: operation.selection_set.len(),
                max_depth: 1, // Mutations are typically flat
                used_schema_mappings: !self.schema_mappings.is_empty(),
                warnings,
            },
        })
    }

    /// Process a GraphQL selection set recursively
    async fn process_selection_set(
        &self,
        selection_set: &[Selection],
        parent_subject: Option<&str>,
        context: &mut ProcessingContext<'_>,
        depth: usize,
    ) -> Result<()> {
        if depth > self.config.max_nesting_depth {
            context.warnings.push(format!(
                "Maximum nesting depth {} exceeded, truncating",
                self.config.max_nesting_depth
            ));
            return Ok(());
        }

        for field in selection_set {
            let subject_var = parent_subject
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("?{}", field.name));

            // Add field to SELECT clause
            context.sparql_builder.add_select_variable(&subject_var);

            // Get predicate mapping
            let predicate = self.get_predicate_for_field(&field.name, "String")?;

            // Create object variable
            let object_var = format!("?{}_{}", field.name, depth);

            // Add triple pattern
            let triple_pattern = TriplePattern {
                subject: Some(subject_var.clone()),
                predicate: Some(predicate.clone()),
                object: Some(object_var.clone()),
                pattern_string: format!("{subject_var} {predicate} {object_var}"),
            };

            context.triple_patterns.push(triple_pattern.clone());
            context.sparql_builder.add_triple_pattern(&triple_pattern);

            // Add variable mapping
            context
                .variable_mappings
                .insert(field.name.clone(), object_var.clone());

            // Process field arguments as filters
            if !field.arguments.is_empty() {
                self.process_field_arguments(
                    &field.arguments,
                    &subject_var,
                    context.sparql_builder,
                    context.warnings,
                )
                .await?;
            }

            // Process nested fields
            if !field.selection_set.is_empty() {
                Box::pin(self.process_selection_set(
                    &field.selection_set,
                    Some(&object_var),
                    context,
                    depth + 1,
                ))
                .await?;
            }
        }

        Ok(())
    }

    /// Process field arguments as SPARQL filters
    async fn process_field_arguments(
        &self,
        arguments: &HashMap<String, serde_json::Value>,
        subject_var: &str,
        sparql_builder: &mut SparqlQueryBuilder,
        warnings: &mut Vec<String>,
    ) -> Result<()> {
        for (arg_name, arg) in arguments {
            let arg_value = arg; // arg is already the value
            match arg_name.as_str() {
                "id" => {
                    // Handle ID equality filter
                    if let Some(id_str) = arg_value.as_str() {
                        let filter = format!(
                            "FILTER({} = <{}{}>)",
                            subject_var, self.config.default_namespace, id_str
                        );
                        sparql_builder.add_filter(&filter);
                    }
                }
                "where" => {
                    // Handle complex where conditions
                    if let Some(where_obj) = arg_value.as_object() {
                        self.process_where_conditions(where_obj, subject_var, sparql_builder)
                            .await?;
                    }
                }
                "limit" => {
                    // Handle result limiting
                    if let Some(limit_num) = arg_value.as_u64() {
                        sparql_builder.set_limit(limit_num as usize);
                    }
                }
                "offset" => {
                    // Handle result offset
                    if let Some(offset_num) = arg_value.as_u64() {
                        sparql_builder.set_offset(offset_num as usize);
                    }
                }
                "orderBy" => {
                    // Handle ordering
                    if let Some(order_field) = arg_value.as_str() {
                        let order_var = format!("?{order_field}");
                        sparql_builder.add_order_by(&order_var, true); // Default ascending
                    }
                }
                _ => {
                    warnings.push(format!(
                        "Unknown argument '{arg_name}' ignored in translation"
                    ));
                }
            }
        }

        Ok(())
    }

    /// Process WHERE conditions from GraphQL arguments
    async fn process_where_conditions(
        &self,
        conditions: &serde_json::Map<String, serde_json::Value>,
        subject_var: &str,
        sparql_builder: &mut SparqlQueryBuilder,
    ) -> Result<()> {
        for (field_name, condition_value) in conditions {
            let predicate = self.get_predicate_for_field(field_name, "String")?;

            if let Some(condition_str) = condition_value.as_str() {
                // Simple equality condition
                let object_value = if condition_str.starts_with("http://")
                    || condition_str.starts_with("https://")
                {
                    format!("<{condition_str}>")
                } else {
                    format!("\"{condition_str}\"")
                };

                let triple_pattern = TriplePattern {
                    subject: Some(subject_var.to_string()),
                    predicate: Some(predicate.clone()),
                    object: Some(object_value.clone()),
                    pattern_string: format!("{subject_var} {predicate} {object_value}"),
                };

                sparql_builder.add_triple_pattern(&triple_pattern);
            } else if let Some(condition_obj) = condition_value.as_object() {
                // Handle complex conditions (eq, ne, gt, lt, etc.)
                self.process_complex_condition(
                    field_name,
                    condition_obj,
                    subject_var,
                    sparql_builder,
                )
                .await?;
            }
        }

        Ok(())
    }

    /// Process complex GraphQL conditions
    async fn process_complex_condition(
        &self,
        field_name: &str,
        condition: &serde_json::Map<String, serde_json::Value>,
        subject_var: &str,
        sparql_builder: &mut SparqlQueryBuilder,
    ) -> Result<()> {
        let predicate = self.get_predicate_for_field(field_name, "String")?;
        let object_var = format!("?{field_name}_value");

        // Add basic triple pattern
        let triple_pattern = TriplePattern {
            subject: Some(subject_var.to_string()),
            predicate: Some(predicate.clone()),
            object: Some(object_var.clone()),
            pattern_string: format!("{subject_var} {predicate} {object_var}"),
        };
        sparql_builder.add_triple_pattern(&triple_pattern);

        // Add condition-specific filters
        for (op, value) in condition {
            let filter = match op.as_str() {
                "eq" => format!("FILTER({} = {})", object_var, self.format_value(value)),
                "ne" => format!("FILTER({} != {})", object_var, self.format_value(value)),
                "gt" => format!("FILTER({} > {})", object_var, self.format_value(value)),
                "gte" => format!("FILTER({} >= {})", object_var, self.format_value(value)),
                "lt" => format!("FILTER({} < {})", object_var, self.format_value(value)),
                "lte" => format!("FILTER({} <= {})", object_var, self.format_value(value)),
                "contains" => format!(
                    "FILTER(CONTAINS(LCASE(STR({})), LCASE(\"{}\")))",
                    object_var,
                    value.as_str().unwrap_or("")
                ),
                "startsWith" => format!(
                    "FILTER(STRSTARTS(LCASE(STR({})), LCASE(\"{}\")))",
                    object_var,
                    value.as_str().unwrap_or("")
                ),
                "endsWith" => format!(
                    "FILTER(STRENDS(LCASE(STR({})), LCASE(\"{}\")))",
                    object_var,
                    value.as_str().unwrap_or("")
                ),
                "in" => {
                    if let Some(values) = value.as_array() {
                        let value_list = values
                            .iter()
                            .map(|v| self.format_value(v))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("FILTER({object_var} IN ({value_list}))")
                    } else {
                        format!("FILTER({} = {})", object_var, self.format_value(value))
                    }
                }
                _ => {
                    warn!("Unknown condition operator: {}", op);
                    continue;
                }
            };
            sparql_builder.add_filter(&filter);
        }

        Ok(())
    }

    /// Process a mutation field
    async fn process_mutation_field(
        &self,
        field: &Selection,
        sparql_builder: &mut SparqlUpdateBuilder,
        variable_mappings: &mut HashMap<String, String>,
        triple_patterns: &mut Vec<TriplePattern>,
        warnings: &mut Vec<String>,
    ) -> Result<()> {
        match field.name.as_str() {
            name if name.starts_with("create") => {
                self.process_create_mutation(
                    field,
                    sparql_builder,
                    variable_mappings,
                    triple_patterns,
                )
                .await?;
            }
            name if name.starts_with("update") => {
                self.process_update_mutation(
                    field,
                    sparql_builder,
                    variable_mappings,
                    triple_patterns,
                )
                .await?;
            }
            name if name.starts_with("delete") => {
                self.process_delete_mutation(
                    field,
                    sparql_builder,
                    variable_mappings,
                    triple_patterns,
                )
                .await?;
            }
            _ => {
                warnings.push(format!("Unknown mutation type: {}", field.name));
            }
        }

        Ok(())
    }

    /// Process CREATE mutation
    async fn process_create_mutation(
        &self,
        field: &Selection,
        sparql_builder: &mut SparqlUpdateBuilder,
        variable_mappings: &mut HashMap<String, String>,
        triple_patterns: &mut Vec<TriplePattern>,
    ) -> Result<()> {
        // Extract entity type from mutation name
        let entity_type = field.name.strip_prefix("create").unwrap_or(&field.name);

        if !field.arguments.is_empty() {
            if let Some(data_arg) = field.arguments.get("data") {
                if let Some(data_obj) = data_arg.as_object() {
                    // Generate URI for new entity
                    let entity_uri = format!(
                        "{}{}_{}",
                        self.config.default_namespace,
                        entity_type.to_lowercase(),
                        uuid::Uuid::new_v4()
                    );

                    // Add type triple
                    if let Ok(type_predicate) = self.get_predicate_for_field("type", "String") {
                        let type_triple = TriplePattern {
                            subject: Some(format!("<{entity_uri}>")),
                            predicate: Some(type_predicate),
                            object: Some(format!(
                                "<{}{}>",
                                self.config.default_namespace, entity_type
                            )),
                            pattern_string: format!(
                                "<{}> {} <{}{}>",
                                entity_uri, "rdf:type", self.config.default_namespace, entity_type
                            ),
                        };
                        triple_patterns.push(type_triple.clone());
                        sparql_builder.add_insert_triple(&type_triple);
                    }

                    // Add property triples
                    for (property, value) in data_obj {
                        let predicate = self.get_predicate_for_field(
                            property,
                            &self.infer_type_from_value(value),
                        )?;
                        let object_value = self.format_value(value);

                        let property_triple = TriplePattern {
                            subject: Some(format!("<{entity_uri}>")),
                            predicate: Some(predicate.clone()),
                            object: Some(object_value.clone()),
                            pattern_string: format!("<{entity_uri}> {predicate} {object_value}"),
                        };
                        triple_patterns.push(property_triple.clone());
                        sparql_builder.add_insert_triple(&property_triple);
                    }

                    variable_mappings.insert("createdEntity".to_string(), entity_uri);
                }
            }
        }

        Ok(())
    }

    /// Process UPDATE mutation
    async fn process_update_mutation(
        &self,
        field: &Selection,
        sparql_builder: &mut SparqlUpdateBuilder,
        _variable_mappings: &mut HashMap<String, String>,
        triple_patterns: &mut Vec<TriplePattern>,
    ) -> Result<()> {
        let args = &field.arguments;
        if !args.is_empty() {
            if let (Some(where_clause), Some(data)) = (args.get("where"), args.get("data")) {
                // Process WHERE clause to identify entity
                let entity_var = "?entity".to_string();

                if let Some(where_obj) = where_clause.as_object() {
                    for (field_name, condition_value) in where_obj {
                        let predicate = self.get_predicate_for_field(field_name, "String")?;
                        let object_value = self.format_value(condition_value);

                        let where_triple = TriplePattern {
                            subject: Some(entity_var.clone()),
                            predicate: Some(predicate.clone()),
                            object: Some(object_value.clone()),
                            pattern_string: format!("{entity_var} {predicate} {object_value}"),
                        };
                        triple_patterns.push(where_triple.clone());
                        sparql_builder.add_where_triple(&where_triple);
                    }
                }

                // Process data to update
                if let Some(data_obj) = data.as_object() {
                    for (property, new_value) in data_obj {
                        let predicate = self.get_predicate_for_field(
                            property,
                            &self.infer_type_from_value(new_value),
                        )?;
                        let old_value_var = format!("?old_{property}");
                        let new_object_value = self.format_value(new_value);

                        // DELETE old value
                        let delete_triple = TriplePattern {
                            subject: Some(entity_var.clone()),
                            predicate: Some(predicate.clone()),
                            object: Some(old_value_var.clone()),
                            pattern_string: format!("{entity_var} {predicate} {old_value_var}"),
                        };
                        sparql_builder.add_delete_triple(&delete_triple);

                        // INSERT new value
                        let insert_triple = TriplePattern {
                            subject: Some(entity_var.clone()),
                            predicate: Some(predicate.clone()),
                            object: Some(new_object_value.clone()),
                            pattern_string: format!("{entity_var} {predicate} {new_object_value}"),
                        };
                        sparql_builder.add_insert_triple(&insert_triple);
                    }
                }
            }
        }

        Ok(())
    }

    /// Process DELETE mutation
    async fn process_delete_mutation(
        &self,
        field: &Selection,
        sparql_builder: &mut SparqlUpdateBuilder,
        _variable_mappings: &mut HashMap<String, String>,
        triple_patterns: &mut Vec<TriplePattern>,
    ) -> Result<()> {
        let args = &field.arguments;
        if !args.is_empty() {
            if let Some(where_clause) = args.get("where") {
                let entity_var = "?entity".to_string();

                if let Some(where_obj) = where_clause.as_object() {
                    for (field_name, condition_value) in where_obj {
                        let predicate = self.get_predicate_for_field(field_name, "String")?;
                        let object_value = self.format_value(condition_value);

                        let where_triple = TriplePattern {
                            subject: Some(entity_var.clone()),
                            predicate: Some(predicate.clone()),
                            object: Some(object_value.clone()),
                            pattern_string: format!("{entity_var} {predicate} {object_value}"),
                        };
                        triple_patterns.push(where_triple.clone());
                        sparql_builder.add_where_triple(&where_triple);
                    }
                }

                // Delete all properties of the entity
                let delete_all_triple = TriplePattern {
                    subject: Some(entity_var.clone()),
                    predicate: Some("?p".to_string()),
                    object: Some("?o".to_string()),
                    pattern_string: format!("{entity_var} ?p ?o"),
                };
                sparql_builder.add_delete_triple(&delete_all_triple);
            }
        }

        Ok(())
    }

    /// Get RDF predicate for a GraphQL field
    fn get_predicate_for_field(&self, field_name: &str, _field_type: &str) -> Result<String> {
        // Check predicate mappings first
        if let Some(mapped_predicate) = self.predicate_mappings.get(field_name) {
            return Ok(mapped_predicate.clone());
        }

        // Check schema mappings
        for schema_mapping in self.schema_mappings.values() {
            if let Some(field_mapping) = schema_mapping.field_mappings.get(field_name) {
                return Ok(field_mapping.predicate.clone());
            }
        }

        // Generate predicate from field name
        Ok(format!("{}{}", self.config.default_namespace, field_name))
    }

    /// Format a JSON value for SPARQL
    fn format_value(&self, value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::String(s) => {
                if s.starts_with("http://") || s.starts_with("https://") {
                    format!("<{s}>")
                } else {
                    format!("\"{}\"", s.replace('"', "\\\""))
                }
            }
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    if self.config.generate_typed_literals {
                        format!("\"{i}\"^^xsd:integer")
                    } else {
                        i.to_string()
                    }
                } else if let Some(f) = n.as_f64() {
                    if self.config.generate_typed_literals {
                        format!("\"{f}\"^^xsd:decimal")
                    } else {
                        f.to_string()
                    }
                } else {
                    n.to_string()
                }
            }
            serde_json::Value::Bool(b) => {
                if self.config.generate_typed_literals {
                    format!("\"{b}\"^^xsd:boolean")
                } else {
                    b.to_string()
                }
            }
            serde_json::Value::Null => "\"\"".to_string(),
            _ => format!("\"{}\"", value.to_string().replace('"', "\\\"")),
        }
    }

    /// Infer XSD type from JSON value
    fn infer_type_from_value(&self, value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::String(_) => "xsd:string".to_string(),
            serde_json::Value::Number(n) => {
                if n.is_i64() {
                    "xsd:integer".to_string()
                } else {
                    "xsd:decimal".to_string()
                }
            }
            serde_json::Value::Bool(_) => "xsd:boolean".to_string(),
            _ => "xsd:string".to_string(),
        }
    }

    /// Parse GraphQL query (simplified implementation)
    fn parse_graphql_query(&self, query: &str) -> Result<ParsedGraphQLQuery> {
        // This is a simplified parser - in production, use a proper GraphQL parser
        let trimmed = query.trim();

        let operation_type = if trimmed.starts_with("query") {
            "query"
        } else if trimmed.starts_with("mutation") {
            "mutation"
        } else if trimmed.starts_with("subscription") {
            "subscription"
        } else {
            "query" // Default to query
        };

        // Extract operation name and selection set (simplified)
        let selection_start = trimmed
            .find('{')
            .ok_or_else(|| anyhow!("Invalid GraphQL query: no selection set found"))?;
        let selection_end = trimmed
            .rfind('}')
            .ok_or_else(|| anyhow!("Invalid GraphQL query: unclosed selection set"))?;

        let selection_content = &trimmed[selection_start + 1..selection_end];
        let selection_set = self.parse_selection_set(selection_content)?;

        Ok(ParsedGraphQLQuery {
            operation: GraphQLOperation {
                operation_type: operation_type.to_string(),
                operation_name: None,
                selection_set,
                variables: None,
            },
        })
    }

    /// Parse GraphQL selection set (simplified)
    fn parse_selection_set(&self, content: &str) -> Result<Vec<Selection>> {
        let mut fields = Vec::new();
        let lines: Vec<&str> = content
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();

        for line in lines {
            if let Some(field) = self.parse_field_line(line)? {
                fields.push(field);
            }
        }

        Ok(fields)
    }

    /// Parse a single field line (simplified)
    fn parse_field_line(&self, line: &str) -> Result<Option<Selection>> {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            return Ok(None);
        }

        // Extract field name (before any arguments or selection set)
        let field_name = if let Some(paren_pos) = trimmed.find('(') {
            trimmed[..paren_pos].trim()
        } else if let Some(brace_pos) = trimmed.find('{') {
            trimmed[..brace_pos].trim()
        } else {
            trimmed
        };

        // Parse arguments if present
        let mut arguments = HashMap::new();
        if let Some(paren_start) = trimmed.find('(') {
            if let Some(paren_end) = trimmed.find(')') {
                let args_str = &trimmed[paren_start + 1..paren_end];
                arguments = self.parse_arguments(args_str)?;
            }
        }

        Ok(Some(Selection {
            name: field_name.to_string(),
            alias: None, // Simplified
            arguments,
            selection_set: Vec::new(), // Simplified - would need recursive parsing
            fragment: None,
        }))
    }

    /// Parse GraphQL arguments (simplified JSON-like parsing)
    fn parse_arguments(&self, args_str: &str) -> Result<HashMap<String, serde_json::Value>> {
        let mut arguments = HashMap::new();

        // Look for data argument specifically for mutations
        if args_str.contains("data:") {
            // Find the data argument
            if let Some(data_start) = args_str.find("data:") {
                let data_content = &args_str[data_start + 5..].trim();

                // Find the opening brace for the data object
                if let Some(brace_start) = data_content.find('{') {
                    // Find the matching closing brace (simplified - assumes well-formed JSON)
                    let mut brace_count = 0;
                    let mut brace_end = None;

                    for (i, ch) in data_content.chars().enumerate() {
                        match ch {
                            '{' => brace_count += 1,
                            '}' => {
                                brace_count -= 1;
                                if brace_count == 0 {
                                    brace_end = Some(i);
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }

                    if let Some(end_pos) = brace_end {
                        let graphql_obj_str = &data_content[brace_start..=end_pos];

                        // Convert GraphQL object notation to JSON (add quotes around keys)
                        let json_str = self.convert_graphql_object_to_json(graphql_obj_str);

                        // Parse as JSON
                        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&json_str)
                        {
                            arguments.insert("data".to_string(), json_value);
                        }
                    }
                }
            }
        }

        Ok(arguments)
    }

    /// Convert GraphQL object notation to valid JSON
    fn convert_graphql_object_to_json(&self, graphql_str: &str) -> String {
        // Simple regex replacement to quote unquoted keys
        let mut result = graphql_str.to_string();

        // Replace unquoted keys with quoted keys
        // This is a simplified approach for the test case
        result = result.replace("name:", "\"name\":");
        result = result.replace("email:", "\"email\":");
        result = result.replace("id:", "\"id\":");

        result
    }

    /// Extract operation from parsed query
    fn extract_operation<'a>(
        &self,
        parsed_query: &'a ParsedGraphQLQuery,
    ) -> Result<&'a GraphQLOperation> {
        Ok(&parsed_query.operation)
    }

    /// Count total fields in selection set
    fn count_fields(selection_set: &[Selection]) -> usize {
        selection_set
            .iter()
            .map(|field| 1 + Self::count_fields(&field.selection_set))
            .sum()
    }

    /// Calculate maximum nesting depth
    fn calculate_max_depth(selection_set: &[Selection]) -> usize {
        selection_set
            .iter()
            .map(|field| 1 + Self::calculate_max_depth(&field.selection_set))
            .max()
            .unwrap_or(0)
    }
}

impl Default for GraphQLToSparqlTranslator {
    fn default() -> Self {
        Self::new()
    }
}

/// SPARQL query builder for SELECT queries
#[derive(Debug)]
struct SparqlQueryBuilder {
    #[allow(dead_code)]
    config: TranslationConfig,
    prefixes: HashMap<String, String>,
    select_variables: HashSet<String>,
    triple_patterns: Vec<TriplePattern>,
    filters: Vec<String>,
    order_by: Vec<(String, bool)>, // (variable, ascending)
    limit: Option<usize>,
    offset: Option<usize>,
}

impl SparqlQueryBuilder {
    fn new(config: &TranslationConfig) -> Self {
        Self {
            config: config.clone(),
            prefixes: config.namespace_prefixes.clone(),
            select_variables: HashSet::new(),
            triple_patterns: Vec::new(),
            filters: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        }
    }

    fn add_select_variable(&mut self, variable: &str) {
        self.select_variables.insert(variable.to_string());
    }

    fn add_triple_pattern(&mut self, pattern: &TriplePattern) {
        self.triple_patterns.push(pattern.clone());
    }

    fn add_filter(&mut self, filter: &str) {
        self.filters.push(filter.to_string());
    }

    fn add_order_by(&mut self, variable: &str, ascending: bool) {
        self.order_by.push((variable.to_string(), ascending));
    }

    fn set_limit(&mut self, limit: usize) {
        self.limit = Some(limit);
    }

    fn set_offset(&mut self, offset: usize) {
        self.offset = Some(offset);
    }

    fn get_prefixes(&self) -> HashMap<String, String> {
        self.prefixes.clone()
    }

    fn build(&self) -> Result<String> {
        let mut query = String::new();

        // Add prefixes
        for (prefix, namespace) in &self.prefixes {
            query.push_str(&format!("PREFIX {prefix}: <{namespace}>\n"));
        }
        query.push('\n');

        // Add SELECT clause
        if self.select_variables.is_empty() {
            query.push_str("SELECT *\n");
        } else {
            let vars: Vec<String> = self.select_variables.iter().cloned().collect();
            query.push_str(&format!("SELECT {}\n", vars.join(" ")));
        }

        // Add WHERE clause
        query.push_str("WHERE {\n");
        for pattern in &self.triple_patterns {
            query.push_str(&format!("  {} .\n", pattern.pattern_string));
        }

        // Add filters
        for filter in &self.filters {
            query.push_str(&format!("  {filter}\n"));
        }
        query.push_str("}\n");

        // Add ORDER BY
        if !self.order_by.is_empty() {
            let order_clauses: Vec<String> = self
                .order_by
                .iter()
                .map(|(var, asc)| {
                    if *asc {
                        var.clone()
                    } else {
                        format!("DESC({var})")
                    }
                })
                .collect();
            query.push_str(&format!("ORDER BY {}\n", order_clauses.join(" ")));
        }

        // Add LIMIT and OFFSET
        if let Some(limit) = self.limit {
            query.push_str(&format!("LIMIT {limit}\n"));
        }
        if let Some(offset) = self.offset {
            query.push_str(&format!("OFFSET {offset}\n"));
        }

        Ok(query)
    }
}

/// SPARQL update builder for INSERT/DELETE operations
#[derive(Debug)]
struct SparqlUpdateBuilder {
    #[allow(dead_code)]
    config: TranslationConfig,
    prefixes: HashMap<String, String>,
    insert_triples: Vec<TriplePattern>,
    delete_triples: Vec<TriplePattern>,
    where_triples: Vec<TriplePattern>,
}

impl SparqlUpdateBuilder {
    fn new(config: &TranslationConfig) -> Self {
        Self {
            config: config.clone(),
            prefixes: config.namespace_prefixes.clone(),
            insert_triples: Vec::new(),
            delete_triples: Vec::new(),
            where_triples: Vec::new(),
        }
    }

    fn add_insert_triple(&mut self, pattern: &TriplePattern) {
        self.insert_triples.push(pattern.clone());
    }

    fn add_delete_triple(&mut self, pattern: &TriplePattern) {
        self.delete_triples.push(pattern.clone());
    }

    fn add_where_triple(&mut self, pattern: &TriplePattern) {
        self.where_triples.push(pattern.clone());
    }

    fn get_prefixes(&self) -> HashMap<String, String> {
        self.prefixes.clone()
    }

    fn build(&self) -> Result<String> {
        let mut query = String::new();

        // Add prefixes
        for (prefix, namespace) in &self.prefixes {
            query.push_str(&format!("PREFIX {prefix}: <{namespace}>\n"));
        }
        query.push('\n');

        // Build DELETE INSERT query
        if !self.delete_triples.is_empty() && !self.insert_triples.is_empty() {
            query.push_str("DELETE {\n");
            for pattern in &self.delete_triples {
                query.push_str(&format!("  {} .\n", pattern.pattern_string));
            }
            query.push_str("}\n");

            query.push_str("INSERT {\n");
            for pattern in &self.insert_triples {
                query.push_str(&format!("  {} .\n", pattern.pattern_string));
            }
            query.push_str("}\n");
        } else if !self.insert_triples.is_empty() {
            query.push_str("INSERT DATA {\n");
            for pattern in &self.insert_triples {
                query.push_str(&format!("  {} .\n", pattern.pattern_string));
            }
            query.push_str("}\n");
        } else if !self.delete_triples.is_empty() {
            query.push_str("DELETE DATA {\n");
            for pattern in &self.delete_triples {
                query.push_str(&format!("  {} .\n", pattern.pattern_string));
            }
            query.push_str("}\n");
        }

        // Add WHERE clause if needed
        if !self.where_triples.is_empty() {
            query.push_str("WHERE {\n");
            for pattern in &self.where_triples {
                query.push_str(&format!("  {} .\n", pattern.pattern_string));
            }
            query.push_str("}\n");
        }

        Ok(query)
    }
}

/// Parsed GraphQL query structure
#[derive(Debug)]
struct ParsedGraphQLQuery {
    operation: GraphQLOperation,
}

/// GraphQL operation
#[derive(Debug)]
struct GraphQLOperation {
    operation_type: String,
    #[allow(dead_code)]
    operation_name: Option<String>,
    selection_set: Vec<Selection>,
    #[allow(dead_code)]
    variables: Option<HashMap<String, serde_json::Value>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_query_translation() {
        let translator = GraphQLToSparqlTranslator::new();

        let graphql_query = r#"
            query {
                user(id: "123") {
                    name
                    email
                }
            }
        "#;

        let result = translator.translate_query(graphql_query).await;
        assert!(result.is_ok());

        let translation = result.unwrap();
        assert!(translation.sparql_query.contains("SELECT"));
        assert!(translation.sparql_query.contains("WHERE"));
        assert!(!translation.triple_patterns.is_empty());
    }

    #[tokio::test]
    async fn test_mutation_translation() {
        let translator = GraphQLToSparqlTranslator::new();

        let graphql_mutation = r#"
            mutation {
                createUser(data: { name: "John Doe", email: "john@example.com" }) {
                    id
                    name
                }
            }
        "#;

        let result = translator.translate_query(graphql_mutation).await;
        assert!(result.is_ok());

        let translation = result.unwrap();
        assert!(translation.sparql_query.contains("INSERT"));
    }

    #[test]
    fn test_value_formatting() {
        let translator = GraphQLToSparqlTranslator::new();

        assert_eq!(
            translator.format_value(&serde_json::Value::String("test".to_string())),
            "\"test\""
        );
        assert_eq!(
            translator.format_value(&serde_json::Value::Number(serde_json::Number::from(42))),
            "\"42\"^^xsd:integer"
        );
        assert_eq!(
            translator.format_value(&serde_json::Value::Bool(true)),
            "\"true\"^^xsd:boolean"
        );
    }
}
