// GraphQL to SPARQL Translation Module
// Provides comprehensive translation of GraphQL queries to SPARQL algebra

use crate::algebra::{Algebra, Term, TriplePattern, Variable};
use oxirs_core::model::NamedNode;
use scirs2_core::metrics::MetricsRegistry;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use thiserror::Error;

#[allow(dead_code)]
/// Errors that can occur during GraphQL to SPARQL translation
#[derive(Error, Debug)]
pub enum TranslationError {
    #[error("Unsupported GraphQL operation: {0}")]
    UnsupportedOperation(String),

    #[error("Invalid field mapping: {0}")]
    InvalidFieldMapping(String),

    #[error("Unknown GraphQL type: {0}")]
    UnknownType(String),

    #[error("Fragment not found: {0}")]
    FragmentNotFound(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Schema mapping error: {0}")]
    SchemaMappingError(String),

    #[error("Variable resolution error: {0}")]
    VariableResolutionError(String),

    #[error("Directive processing error: {0}")]
    DirectiveError(String),

    #[error("Nested query too deep: {0}")]
    QueryTooDeep(usize),

    #[error("Translation failed: {0}")]
    TranslationFailed(String),
}

/// Result type for GraphQL translation operations
pub type TranslationResult<T> = std::result::Result<T, TranslationError>;

/// GraphQL operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphQLOperationType {
    Query,
    Mutation,
    Subscription,
}

/// Simplified GraphQL field representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLField {
    pub name: String,
    pub alias: Option<String>,
    pub arguments: HashMap<String, GraphQLValue>,
    pub directives: Vec<GraphQLDirective>,
    pub selection_set: Vec<GraphQLSelection>,
}

/// GraphQL selection types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphQLSelection {
    Field(GraphQLField),
    FragmentSpread {
        name: String,
        directives: Vec<GraphQLDirective>,
    },
    InlineFragment {
        type_condition: Option<String>,
        directives: Vec<GraphQLDirective>,
        selection_set: Vec<GraphQLSelection>,
    },
}

/// GraphQL directive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLDirective {
    pub name: String,
    pub arguments: HashMap<String, GraphQLValue>,
}

/// GraphQL value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphQLValue {
    Null,
    Int(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Enum(String),
    List(Vec<GraphQLValue>),
    Object(HashMap<String, GraphQLValue>),
    Variable(String),
}

/// GraphQL operation (query/mutation/subscription)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLOperation {
    pub operation_type: GraphQLOperationType,
    pub name: Option<String>,
    pub variables: HashMap<String, GraphQLVariableDefinition>,
    pub directives: Vec<GraphQLDirective>,
    pub selection_set: Vec<GraphQLSelection>,
}

/// GraphQL variable definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLVariableDefinition {
    pub var_type: String,
    pub default_value: Option<GraphQLValue>,
}

/// GraphQL fragment definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLFragment {
    pub name: String,
    pub type_condition: String,
    pub directives: Vec<GraphQLDirective>,
    pub selection_set: Vec<GraphQLSelection>,
}

/// GraphQL document containing operations and fragments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLDocument {
    pub operations: Vec<GraphQLOperation>,
    pub fragments: HashMap<String, GraphQLFragment>,
}

/// Schema mapping configuration for GraphQL to RDF translation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMapping {
    /// Map GraphQL type names to RDF class URIs
    pub type_to_class: HashMap<String, String>,

    /// Map GraphQL field names to RDF property URIs
    pub field_to_property: HashMap<String, String>,

    /// RDF namespace prefixes
    pub prefixes: HashMap<String, String>,

    /// Root type for queries (default: "Query")
    pub query_root_type: String,

    /// Root type for mutations (default: "Mutation")
    pub mutation_root_type: String,

    /// Default RDF type property (default: rdf:type)
    pub rdf_type_property: String,

    /// Enable automatic camelCase to snake_case conversion
    pub auto_case_conversion: bool,
}

impl Default for SchemaMapping {
    fn default() -> Self {
        let mut prefixes = HashMap::new();
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );

        Self {
            type_to_class: HashMap::new(),
            field_to_property: HashMap::new(),
            prefixes,
            query_root_type: "Query".to_string(),
            mutation_root_type: "Mutation".to_string(),
            rdf_type_property: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            auto_case_conversion: true,
        }
    }
}

/// Configuration for GraphQL to SPARQL translation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslatorConfig {
    /// Schema mapping configuration
    pub schema_mapping: SchemaMapping,

    /// Maximum query depth to prevent overly complex translations
    pub max_query_depth: usize,

    /// Enable query optimization after translation
    pub enable_optimization: bool,

    /// Generate SPARQL comments with GraphQL source
    pub generate_comments: bool,

    /// Handle @include and @skip directives
    pub process_directives: bool,
}

impl Default for TranslatorConfig {
    fn default() -> Self {
        Self {
            schema_mapping: SchemaMapping::default(),
            max_query_depth: 10,
            enable_optimization: true,
            generate_comments: true,
            process_directives: true,
        }
    }
}

/// Translation context for tracking state during translation
#[derive(Debug, Clone)]
struct TranslationContext {
    /// Current query depth
    depth: usize,

    /// Variable counter for generating unique SPARQL variables
    var_counter: usize,

    /// Fragment definitions available for expansion
    fragments: HashMap<String, GraphQLFragment>,

    /// GraphQL variable values
    variables: HashMap<String, GraphQLValue>,

    /// Current subject variable
    current_subject: Option<Variable>,

    /// Collected SPARQL variables
    sparql_variables: HashSet<Variable>,

    /// Collected triple patterns (as algebra) - reserved for future optimization
    #[allow(dead_code)]
    patterns: Vec<Algebra>,
}

impl TranslationContext {
    fn new(
        fragments: HashMap<String, GraphQLFragment>,
        variables: HashMap<String, GraphQLValue>,
    ) -> Self {
        Self {
            depth: 0,
            var_counter: 0,
            fragments,
            variables,
            current_subject: None,
            sparql_variables: HashSet::new(),
            patterns: Vec::new(),
        }
    }

    fn next_var(&mut self, prefix: &str) -> Variable {
        self.var_counter += 1;
        let var = Variable::new(format!("{}{}", prefix, self.var_counter))
            .expect("Variable name should be valid");
        self.sparql_variables.insert(var.clone());
        var
    }

    fn enter_scope(&mut self) -> TranslationResult<()> {
        self.depth += 1;
        Ok(())
    }

    fn exit_scope(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }
}

/// Statistics for translation operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TranslationStats {
    pub queries_translated: usize,
    pub mutations_translated: usize,
    pub fields_translated: usize,
    pub fragments_expanded: usize,
    pub directives_processed: usize,
    pub average_query_depth: f64,
    pub translation_errors: usize,
}

/// Main GraphQL to SPARQL translator
pub struct GraphQLTranslator {
    config: TranslatorConfig,
    stats: TranslationStats,
    /// Metrics registry - reserved for future instrumentation
    #[allow(dead_code)]
    metrics: Arc<MetricsRegistry>,
}

impl GraphQLTranslator {
    /// Create a new GraphQL translator with default configuration
    pub fn new() -> Self {
        Self {
            config: TranslatorConfig::default(),
            stats: TranslationStats::default(),
            metrics: Arc::new(MetricsRegistry::new()),
        }
    }

    /// Create a new GraphQL translator with custom configuration
    pub fn with_config(config: TranslatorConfig) -> Self {
        Self {
            config,
            stats: TranslationStats::default(),
            metrics: Arc::new(MetricsRegistry::new()),
        }
    }

    /// Translate a GraphQL document to SPARQL algebra
    pub fn translate_document(
        &mut self,
        document: GraphQLDocument,
    ) -> TranslationResult<Vec<Algebra>> {
        let mut algebras = Vec::new();

        for operation in document.operations {
            let algebra = self.translate_operation(&operation, &document.fragments)?;
            algebras.push(algebra);
        }

        Ok(algebras)
    }

    /// Translate a single GraphQL operation to SPARQL algebra
    pub fn translate_operation(
        &mut self,
        operation: &GraphQLOperation,
        fragments: &HashMap<String, GraphQLFragment>,
    ) -> TranslationResult<Algebra> {
        let mut context = TranslationContext::new(fragments.clone(), HashMap::new());

        match operation.operation_type {
            GraphQLOperationType::Query => {
                self.stats.queries_translated += 1;
                self.translate_query(operation, &mut context)
            }
            GraphQLOperationType::Mutation => {
                self.stats.mutations_translated += 1;
                self.translate_mutation(operation, &mut context)
            }
            GraphQLOperationType::Subscription => Err(TranslationError::UnsupportedOperation(
                "Subscriptions are not yet supported".to_string(),
            )),
        }
    }

    /// Translate a GraphQL query to SPARQL SELECT algebra
    fn translate_query(
        &mut self,
        operation: &GraphQLOperation,
        context: &mut TranslationContext,
    ) -> TranslationResult<Algebra> {
        context.enter_scope()?;

        // Process selection set
        let patterns = self.translate_selection_set(&operation.selection_set, context)?;

        // Combine all patterns into a single BGP
        let bgp = if patterns.is_empty() {
            Algebra::Bgp(vec![])
        } else if patterns.len() == 1 {
            patterns.into_iter().next().unwrap()
        } else {
            // Join all patterns
            patterns
                .into_iter()
                .reduce(|acc, pattern| Algebra::Join {
                    left: Box::new(acc),
                    right: Box::new(pattern),
                })
                .unwrap()
        };

        // Project the variables collected during translation
        let variables: Vec<Variable> = context.sparql_variables.iter().cloned().collect();
        let result = Algebra::Project {
            pattern: Box::new(bgp),
            variables,
        };

        context.exit_scope();
        Ok(result)
    }

    /// Translate a GraphQL mutation to SPARQL UPDATE algebra
    fn translate_mutation(
        &mut self,
        _operation: &GraphQLOperation,
        _context: &mut TranslationContext,
    ) -> TranslationResult<Algebra> {
        // For now, return a basic pattern
        // Full mutation support would require UPDATE algebra extensions
        Err(TranslationError::UnsupportedOperation(
            "Mutations require SPARQL UPDATE support".to_string(),
        ))
    }

    /// Translate a GraphQL selection set to SPARQL patterns
    fn translate_selection_set(
        &mut self,
        selections: &[GraphQLSelection],
        context: &mut TranslationContext,
    ) -> TranslationResult<Vec<Algebra>> {
        if context.depth > self.config.max_query_depth {
            return Err(TranslationError::QueryTooDeep(context.depth));
        }

        let mut patterns = Vec::new();

        for selection in selections {
            match selection {
                GraphQLSelection::Field(field) => {
                    let pattern = self.translate_field(field, context)?;
                    patterns.push(pattern);
                    self.stats.fields_translated += 1;
                }
                GraphQLSelection::FragmentSpread { name, directives } => {
                    if self.config.process_directives
                        && self.should_skip_by_directives(directives, context)?
                    {
                        continue;
                    }

                    let fragment = context
                        .fragments
                        .get(name)
                        .ok_or_else(|| TranslationError::FragmentNotFound(name.clone()))?;

                    // Clone the selection set to avoid borrow checker issues
                    let selection_set = fragment.selection_set.clone();
                    let fragment_patterns =
                        self.translate_selection_set(&selection_set, context)?;
                    patterns.extend(fragment_patterns);
                    self.stats.fragments_expanded += 1;
                }
                GraphQLSelection::InlineFragment {
                    type_condition,
                    directives,
                    selection_set,
                } => {
                    if self.config.process_directives
                        && self.should_skip_by_directives(directives, context)?
                    {
                        continue;
                    }

                    // Add type filter if type condition is present
                    if let Some(type_name) = type_condition {
                        if let Some(class_uri) =
                            self.config.schema_mapping.type_to_class.get(type_name)
                        {
                            // Add rdf:type filter pattern
                            let subject = context
                                .current_subject
                                .clone()
                                .unwrap_or_else(|| context.next_var("subject"));

                            let type_property =
                                NamedNode::new(&self.config.schema_mapping.rdf_type_property)
                                    .expect("Invalid RDF type property URI");
                            let class_node = NamedNode::new(class_uri).expect("Invalid class URI");

                            let type_pattern = Algebra::Bgp(vec![TriplePattern::new(
                                Term::Variable(subject.clone()),
                                Term::Iri(type_property),
                                Term::Iri(class_node),
                            )]);
                            patterns.push(type_pattern);
                        }
                    }

                    let inline_patterns = self.translate_selection_set(selection_set, context)?;
                    patterns.extend(inline_patterns);
                }
            }
        }

        Ok(patterns)
    }

    /// Translate a GraphQL field to SPARQL pattern
    fn translate_field(
        &mut self,
        field: &GraphQLField,
        context: &mut TranslationContext,
    ) -> TranslationResult<Algebra> {
        // Check directives
        if self.config.process_directives
            && self.should_skip_by_directives(&field.directives, context)?
        {
            return Ok(Algebra::Bgp(vec![]));
        }

        // Get or create subject variable
        let subject = context
            .current_subject
            .clone()
            .unwrap_or_else(|| context.next_var("subject"));

        // Map field name to RDF property
        let property_uri = self.map_field_to_property(&field.name)?;

        // Create object variable for this field
        let object_var_name = field.alias.as_ref().unwrap_or(&field.name);
        let object = context.next_var(object_var_name);

        // Create basic triple pattern
        let property_node = NamedNode::new(&property_uri).expect("Invalid property URI");

        let triple_pattern = Algebra::Bgp(vec![TriplePattern::new(
            Term::Variable(subject.clone()),
            Term::Iri(property_node),
            Term::Variable(object.clone()),
        )]);

        // Handle nested selections
        if !field.selection_set.is_empty() {
            let old_subject = context.current_subject.replace(object.clone());
            context.enter_scope()?;

            let nested_patterns = self.translate_selection_set(&field.selection_set, context)?;

            context.exit_scope();
            context.current_subject = old_subject;

            // Join the triple pattern with nested patterns
            if nested_patterns.is_empty() {
                return Ok(triple_pattern);
            }

            let nested_algebra = nested_patterns
                .into_iter()
                .reduce(|acc, pattern| Algebra::Join {
                    left: Box::new(acc),
                    right: Box::new(pattern),
                })
                .unwrap();

            return Ok(Algebra::Join {
                left: Box::new(triple_pattern),
                right: Box::new(nested_algebra),
            });
        }

        // Handle arguments as filters
        if !field.arguments.is_empty() {
            let filters = self.translate_arguments(&field.arguments, &object, context)?;
            if !filters.is_empty() {
                // Combine filters with AND logic
                let combined_filter = filters
                    .into_iter()
                    .reduce(|acc, filter| {
                        // For now, just join the filters
                        // Full implementation would create proper FILTER expressions
                        Algebra::Join {
                            left: Box::new(acc),
                            right: Box::new(filter),
                        }
                    })
                    .unwrap();

                return Ok(Algebra::Join {
                    left: Box::new(triple_pattern),
                    right: Box::new(combined_filter),
                });
            }
        }

        Ok(triple_pattern)
    }

    /// Translate GraphQL arguments to SPARQL filters
    fn translate_arguments(
        &mut self,
        arguments: &HashMap<String, GraphQLValue>,
        _object: &Variable,
        _context: &mut TranslationContext,
    ) -> TranslationResult<Vec<Algebra>> {
        let filters = Vec::new();

        for (arg_name, arg_value) in arguments {
            // For now, create placeholder filters
            // Full implementation would create proper SPARQL FILTER expressions
            let _filter_expr = format!("FILTER for {} = {:?}", arg_name, arg_value);
            // filters.push(...) - would add actual filter algebra here
        }

        Ok(filters)
    }

    /// Check if a field should be skipped based on @include/@skip directives
    fn should_skip_by_directives(
        &mut self,
        directives: &[GraphQLDirective],
        context: &mut TranslationContext,
    ) -> TranslationResult<bool> {
        for directive in directives {
            self.stats.directives_processed += 1;

            match directive.name.as_str() {
                "skip" => {
                    if let Some(GraphQLValue::Boolean(should_skip)) = directive.arguments.get("if")
                    {
                        if *should_skip {
                            return Ok(true);
                        }
                    } else if let Some(GraphQLValue::Variable(var_name)) =
                        directive.arguments.get("if")
                    {
                        if let Some(GraphQLValue::Boolean(should_skip)) =
                            context.variables.get(var_name)
                        {
                            if *should_skip {
                                return Ok(true);
                            }
                        }
                    }
                }
                "include" => {
                    if let Some(GraphQLValue::Boolean(should_include)) =
                        directive.arguments.get("if")
                    {
                        if !*should_include {
                            return Ok(true);
                        }
                    } else if let Some(GraphQLValue::Variable(var_name)) =
                        directive.arguments.get("if")
                    {
                        if let Some(GraphQLValue::Boolean(should_include)) =
                            context.variables.get(var_name)
                        {
                            if !*should_include {
                                return Ok(true);
                            }
                        }
                    }
                }
                _ => {
                    // Unknown directive - ignore for now
                }
            }
        }

        Ok(false)
    }

    /// Map a GraphQL field name to an RDF property URI
    fn map_field_to_property(&self, field_name: &str) -> TranslationResult<String> {
        // First check explicit mapping
        if let Some(property_uri) = self.config.schema_mapping.field_to_property.get(field_name) {
            return Ok(property_uri.clone());
        }

        // Try case conversion if enabled
        if self.config.schema_mapping.auto_case_conversion {
            let snake_case = self.camel_to_snake_case(field_name);
            if let Some(property_uri) = self
                .config
                .schema_mapping
                .field_to_property
                .get(&snake_case)
            {
                return Ok(property_uri.clone());
            }
        }

        // Generate a default property URI based on field name
        Ok(format!("http://example.org/property/{}", field_name))
    }

    /// Convert camelCase to snake_case
    /// Handles consecutive uppercase letters by treating them as separate words
    fn camel_to_snake_case(&self, s: &str) -> String {
        let mut result = String::new();
        let chars: Vec<char> = s.chars().collect();

        for (i, &c) in chars.iter().enumerate() {
            if c.is_uppercase() {
                // Add underscore if:
                // 1. Not at the beginning
                // 2. Previous char was lowercase
                // 3. OR next char exists and is lowercase (for consecutive uppercase)
                let should_add_underscore = i > 0
                    && (chars[i - 1].is_lowercase()
                        || (i + 1 < chars.len() && chars[i + 1].is_lowercase()));

                if should_add_underscore && !result.is_empty() && !result.ends_with('_') {
                    result.push('_');
                }
                result.push(c.to_ascii_lowercase());
            } else {
                result.push(c);
            }
        }

        result
    }

    /// Get translation statistics
    pub fn get_stats(&self) -> &TranslationStats {
        &self.stats
    }

    /// Reset translation statistics
    pub fn reset_stats(&mut self) {
        self.stats = TranslationStats::default();
    }

    /// Update schema mapping configuration
    pub fn update_schema_mapping(&mut self, mapping: SchemaMapping) {
        self.config.schema_mapping = mapping;
    }

    /// Add a type mapping (GraphQL type -> RDF class)
    pub fn add_type_mapping(&mut self, graphql_type: String, rdf_class: String) {
        self.config
            .schema_mapping
            .type_to_class
            .insert(graphql_type, rdf_class);
    }

    /// Add a field mapping (GraphQL field -> RDF property)
    pub fn add_field_mapping(&mut self, graphql_field: String, rdf_property: String) {
        self.config
            .schema_mapping
            .field_to_property
            .insert(graphql_field, rdf_property);
    }

    /// Add a namespace prefix
    pub fn add_prefix(&mut self, prefix: String, namespace: String) {
        self.config
            .schema_mapping
            .prefixes
            .insert(prefix, namespace);
    }
}

impl Default for GraphQLTranslator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translator_creation() {
        let translator = GraphQLTranslator::new();
        assert_eq!(translator.stats.queries_translated, 0);
    }

    #[test]
    fn test_camel_to_snake_case() {
        let translator = GraphQLTranslator::new();
        assert_eq!(translator.camel_to_snake_case("firstName"), "first_name");
        assert_eq!(translator.camel_to_snake_case("userName"), "user_name");
        assert_eq!(translator.camel_to_snake_case("id"), "id");
        assert_eq!(
            translator.camel_to_snake_case("HTTPResponse"),
            "http_response"
        );
        assert_eq!(translator.camel_to_snake_case("XMLParser"), "xml_parser");
        assert_eq!(translator.camel_to_snake_case("IOError"), "io_error");
    }

    #[test]
    fn test_schema_mapping_default() {
        let mapping = SchemaMapping::default();
        assert_eq!(mapping.query_root_type, "Query");
        assert_eq!(mapping.mutation_root_type, "Mutation");
        assert!(mapping.prefixes.contains_key("rdf"));
        assert!(mapping.prefixes.contains_key("rdfs"));
        assert!(mapping.prefixes.contains_key("xsd"));
    }

    #[test]
    fn test_add_type_mapping() {
        let mut translator = GraphQLTranslator::new();
        translator.add_type_mapping("User".to_string(), "http://example.org/User".to_string());
        assert_eq!(
            translator.config.schema_mapping.type_to_class.get("User"),
            Some(&"http://example.org/User".to_string())
        );
    }

    #[test]
    fn test_add_field_mapping() {
        let mut translator = GraphQLTranslator::new();
        translator.add_field_mapping(
            "name".to_string(),
            "http://xmlns.com/foaf/0.1/name".to_string(),
        );
        assert_eq!(
            translator
                .config
                .schema_mapping
                .field_to_property
                .get("name"),
            Some(&"http://xmlns.com/foaf/0.1/name".to_string())
        );
    }

    #[test]
    fn test_map_field_to_property_explicit() {
        let mut translator = GraphQLTranslator::new();
        translator.add_field_mapping(
            "email".to_string(),
            "http://xmlns.com/foaf/0.1/mbox".to_string(),
        );

        let property = translator.map_field_to_property("email").unwrap();
        assert_eq!(property, "http://xmlns.com/foaf/0.1/mbox");
    }

    #[test]
    fn test_map_field_to_property_default() {
        let translator = GraphQLTranslator::new();
        let property = translator.map_field_to_property("unknownField").unwrap();
        assert_eq!(property, "http://example.org/property/unknownField");
    }

    #[test]
    fn test_translation_context_variable_generation() {
        let mut context = TranslationContext::new(HashMap::new(), HashMap::new());
        let var1 = context.next_var("test");
        let var2 = context.next_var("test");

        assert_ne!(var1.name(), var2.name());
        assert_eq!(context.sparql_variables.len(), 2);
    }

    #[test]
    fn test_simple_query_translation() {
        let mut translator = GraphQLTranslator::new();

        let operation = GraphQLOperation {
            operation_type: GraphQLOperationType::Query,
            name: Some("GetUsers".to_string()),
            variables: HashMap::new(),
            directives: vec![],
            selection_set: vec![GraphQLSelection::Field(GraphQLField {
                name: "users".to_string(),
                alias: None,
                arguments: HashMap::new(),
                directives: vec![],
                selection_set: vec![
                    GraphQLSelection::Field(GraphQLField {
                        name: "id".to_string(),
                        alias: None,
                        arguments: HashMap::new(),
                        directives: vec![],
                        selection_set: vec![],
                    }),
                    GraphQLSelection::Field(GraphQLField {
                        name: "name".to_string(),
                        alias: None,
                        arguments: HashMap::new(),
                        directives: vec![],
                        selection_set: vec![],
                    }),
                ],
            })],
        };

        let fragments = HashMap::new();
        let result = translator.translate_operation(&operation, &fragments);
        assert!(result.is_ok());
        assert_eq!(translator.stats.queries_translated, 1);
        assert_eq!(translator.stats.fields_translated, 3); // users, id, name
    }

    #[test]
    fn test_skip_directive() {
        let mut translator = GraphQLTranslator::new();
        let mut context = TranslationContext::new(HashMap::new(), HashMap::new());

        let mut skip_args = HashMap::new();
        skip_args.insert("if".to_string(), GraphQLValue::Boolean(true));

        let directives = vec![GraphQLDirective {
            name: "skip".to_string(),
            arguments: skip_args,
        }];

        let should_skip = translator
            .should_skip_by_directives(&directives, &mut context)
            .unwrap();
        assert!(should_skip);
    }

    #[test]
    fn test_include_directive() {
        let mut translator = GraphQLTranslator::new();
        let mut context = TranslationContext::new(HashMap::new(), HashMap::new());

        let mut include_args = HashMap::new();
        include_args.insert("if".to_string(), GraphQLValue::Boolean(false));

        let directives = vec![GraphQLDirective {
            name: "include".to_string(),
            arguments: include_args,
        }];

        let should_skip = translator
            .should_skip_by_directives(&directives, &mut context)
            .unwrap();
        assert!(should_skip);
    }

    #[test]
    fn test_unsupported_subscription() {
        let mut translator = GraphQLTranslator::new();

        let operation = GraphQLOperation {
            operation_type: GraphQLOperationType::Subscription,
            name: Some("OnUserCreated".to_string()),
            variables: HashMap::new(),
            directives: vec![],
            selection_set: vec![],
        };

        let fragments = HashMap::new();
        let result = translator.translate_operation(&operation, &fragments);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TranslationError::UnsupportedOperation(_)
        ));
    }

    #[test]
    fn test_stats_tracking() {
        let mut translator = GraphQLTranslator::new();

        // Simulate some translations
        translator.stats.queries_translated = 5;
        translator.stats.mutations_translated = 3;
        translator.stats.fields_translated = 42;

        let stats = translator.get_stats();
        assert_eq!(stats.queries_translated, 5);
        assert_eq!(stats.mutations_translated, 3);
        assert_eq!(stats.fields_translated, 42);

        translator.reset_stats();
        assert_eq!(translator.stats.queries_translated, 0);
    }
}
