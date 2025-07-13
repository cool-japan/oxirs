//! RDF to GraphQL mapping utilities

use crate::ast::{Document, Field, OperationDefinition, Selection, SelectionSet, Value};
use crate::optimizer::QueryOptimizer;
use crate::schema::{PropertyType, RdfVocabulary};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;

/// Configuration for query translation
#[derive(Debug, Clone)]
pub struct TranslationConfig {
    pub default_limit: usize,
    pub max_limit: usize,
    pub enable_aggregation: bool,
    pub enable_full_text_search: bool,
    pub optimize_queries: bool,
    pub enable_nested_filtering: bool,
    pub enable_unions: bool,
    pub enable_sorting: bool,
    pub max_query_depth: usize,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            default_limit: 100,
            max_limit: 10000,
            enable_aggregation: true,
            enable_full_text_search: false,
            optimize_queries: true,
            enable_nested_filtering: true,
            enable_unions: true,
            enable_sorting: true,
            max_query_depth: 10,
        }
    }
}

/// Context for query translation
#[derive(Debug, Clone)]
pub struct QueryContext {
    pub variables: HashMap<String, Value>,
    pub operation_name: Option<String>,
    pub depth: usize,
    pub fragments: HashMap<String, crate::ast::FragmentDefinition>,
}

impl QueryContext {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            operation_name: None,
            depth: 0,
            fragments: HashMap::new(),
        }
    }
}

/// Maps RDF data to GraphQL types and translates GraphQL queries to SPARQL
pub struct RdfGraphQLMapper {
    namespace_prefixes: HashMap<String, String>,
    vocabulary: Option<RdfVocabulary>,
    config: TranslationConfig,
    optimizer: Option<Arc<QueryOptimizer>>,
    variable_counter: std::cell::RefCell<usize>,
}

impl RdfGraphQLMapper {
    pub fn new() -> Self {
        let mut namespace_prefixes = HashMap::new();

        // Add common namespace prefixes
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
        namespace_prefixes.insert("schema".to_string(), "http://schema.org/".to_string());
        namespace_prefixes.insert(
            "dc".to_string(),
            "http://purl.org/dc/elements/1.1/".to_string(),
        );
        namespace_prefixes.insert(
            "dbo".to_string(),
            "http://dbpedia.org/ontology/".to_string(),
        );
        namespace_prefixes.insert(
            "geo".to_string(),
            "http://www.w3.org/2003/01/geo/wgs84_pos#".to_string(),
        );
        namespace_prefixes.insert(
            "skos".to_string(),
            "http://www.w3.org/2004/02/skos/core#".to_string(),
        );

        Self {
            namespace_prefixes,
            vocabulary: None,
            config: TranslationConfig::default(),
            optimizer: None,
            variable_counter: std::cell::RefCell::new(0),
        }
    }

    pub fn with_vocabulary(mut self, vocabulary: RdfVocabulary) -> Self {
        self.vocabulary = Some(vocabulary);
        self
    }

    pub fn with_config(mut self, config: TranslationConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_optimizer(mut self, optimizer: Arc<QueryOptimizer>) -> Self {
        self.optimizer = Some(optimizer);
        self
    }

    pub fn add_namespace(&mut self, prefix: String, uri: String) {
        self.namespace_prefixes.insert(prefix, uri);
    }

    /// Convert a GraphQL query to SPARQL based on RDF vocabulary mapping
    pub async fn graphql_to_sparql(
        &self,
        document: &Document,
        type_name: &str,
        context: &QueryContext,
    ) -> Result<String> {
        // Use optimizer if available
        if let Some(ref optimizer) = self.optimizer {
            let query_plan = optimizer.get_query_plan(document).await?;
            if self.config.optimize_queries {
                return Ok(query_plan.sparql_query);
            }
        }

        // Reset variable counter for this query
        *self.variable_counter.borrow_mut() = 0;

        let mut sparql_builder = SparqlQueryBuilder::new(&self.namespace_prefixes, &self.config);

        // Extract the operation
        let operation = self.extract_operation(document, &context.operation_name)?;

        // Validate query depth
        let query_depth = self.calculate_query_depth(&operation.selection_set);
        if query_depth > self.config.max_query_depth {
            return Err(anyhow!(
                "Query depth {} exceeds maximum allowed depth {}",
                query_depth,
                self.config.max_query_depth
            ));
        }

        // Handle different operation types
        match operation.operation_type {
            crate::ast::OperationType::Query => {
                self.translate_query_operation(operation, &mut sparql_builder, context, type_name)
            }
            crate::ast::OperationType::Mutation => self.translate_mutation_operation(
                operation,
                &mut sparql_builder,
                context,
                type_name,
            ),
            crate::ast::OperationType::Subscription => {
                Err(anyhow!("Subscription operations not yet supported"))
            }
        }
    }

    fn extract_operation<'a>(
        &self,
        document: &'a Document,
        operation_name: &Option<String>,
    ) -> Result<&'a OperationDefinition> {
        let operations: Vec<_> = document
            .definitions
            .iter()
            .filter_map(|def| match def {
                crate::ast::Definition::Operation(op) => Some(op),
                _ => None,
            })
            .collect();

        match (operations.len(), operation_name) {
            (0, _) => Err(anyhow!("No operations found in document")),
            (1, _) => Ok(operations[0]),
            (_, Some(name)) => operations
                .iter()
                .find(|op| op.name.as_ref() == Some(name))
                .copied()
                .ok_or_else(|| anyhow!("Operation '{}' not found", name)),
            (_, None) => Err(anyhow!(
                "Multiple operations found but no operation name specified"
            )),
        }
    }

    fn calculate_query_depth(&self, selection_set: &SelectionSet) -> usize {
        fn calculate_depth_recursive(selection_set: &SelectionSet, current_depth: usize) -> usize {
            let mut max_depth = current_depth;

            for selection in &selection_set.selections {
                let depth = match selection {
                    Selection::Field(field) => {
                        if let Some(ref nested_selection) = field.selection_set {
                            calculate_depth_recursive(nested_selection, current_depth + 1)
                        } else {
                            current_depth + 1
                        }
                    }
                    Selection::InlineFragment(fragment) => {
                        calculate_depth_recursive(&fragment.selection_set, current_depth)
                    }
                    Selection::FragmentSpread(_) => current_depth + 1, // Simplified
                };
                max_depth = max_depth.max(depth);
            }
            max_depth
        }

        calculate_depth_recursive(selection_set, 0)
    }

    fn translate_query_operation(
        &self,
        operation: &OperationDefinition,
        builder: &mut SparqlQueryBuilder,
        context: &QueryContext,
        type_name: &str,
    ) -> Result<String> {
        // For Query type, analyze the selection set
        if type_name == "Query" {
            self.translate_root_selection_set(&operation.selection_set, builder, context)?;
        } else {
            return Err(anyhow!("Only Query type supported for now"));
        }

        builder.build()
    }

    fn translate_mutation_operation(
        &self,
        operation: &OperationDefinition,
        builder: &mut SparqlQueryBuilder,
        context: &QueryContext,
        type_name: &str,
    ) -> Result<String> {
        // For Mutation type, translate to SPARQL UPDATE operations
        if type_name == "Mutation" {
            self.translate_mutation_selection_set(&operation.selection_set, builder, context)?;
        } else {
            return Err(anyhow!("Only Mutation type supported for mutations"));
        }

        builder.build()
    }

    fn translate_mutation_selection_set(
        &self,
        selection_set: &SelectionSet,
        builder: &mut SparqlQueryBuilder,
        context: &QueryContext,
    ) -> Result<()> {
        for selection in &selection_set.selections {
            match selection {
                Selection::Field(field) => {
                    self.translate_mutation_field(field, builder, context)?;
                }
                Selection::InlineFragment(_) | Selection::FragmentSpread(_) => {
                    // Handle fragments if needed
                }
            }
        }
        Ok(())
    }

    fn translate_mutation_field(
        &self,
        field: &Field,
        builder: &mut SparqlQueryBuilder,
        _context: &QueryContext,
    ) -> Result<()> {
        match field.name.as_str() {
            "insertTriple" => {
                let args: Vec<(String, Value)> = field
                    .arguments
                    .iter()
                    .map(|arg| (arg.name.clone(), arg.value.clone()))
                    .collect();
                let triple = self.extract_triple_from_arguments(&args)?;
                builder.add_insert_data(&triple);
            }
            "insertTriples" => {
                let args: Vec<(String, Value)> = field
                    .arguments
                    .iter()
                    .map(|arg| (arg.name.clone(), arg.value.clone()))
                    .collect();
                let triples = self.extract_triples_from_arguments(&args)?;
                for triple in triples {
                    builder.add_insert_data(&triple);
                }
            }
            "deleteTriple" => {
                let args: Vec<(String, Value)> = field
                    .arguments
                    .iter()
                    .map(|arg| (arg.name.clone(), arg.value.clone()))
                    .collect();
                let triple = self.extract_triple_from_arguments(&args)?;
                builder.add_delete_data(&triple);
            }
            "deleteTriples" => {
                let args: Vec<(String, Value)> = field
                    .arguments
                    .iter()
                    .map(|arg| (arg.name.clone(), arg.value.clone()))
                    .collect();
                let triples = self.extract_triples_from_arguments(&args)?;
                for triple in triples {
                    builder.add_delete_data(&triple);
                }
            }
            "updateTriple" => {
                // Update = Delete old + Insert new
                let args: Vec<(String, Value)> = field
                    .arguments
                    .iter()
                    .map(|arg| (arg.name.clone(), arg.value.clone()))
                    .collect();
                if let Ok(old_triple) = self.extract_old_triple_from_arguments(&args) {
                    builder.add_delete_data(&old_triple);
                }
                if let Ok(new_triple) = self.extract_new_triple_from_arguments(&args) {
                    builder.add_insert_data(&new_triple);
                }
            }
            _ => {
                return Err(anyhow!("Unsupported mutation operation: {}", field.name));
            }
        }
        Ok(())
    }

    fn extract_triple_from_arguments(&self, arguments: &[(String, Value)]) -> Result<String> {
        let mut subject = None;
        let mut predicate = None;
        let mut object = None;

        for (arg_name, arg_value) in arguments {
            match arg_name.as_str() {
                "subject" => {
                    subject = Some(self.format_rdf_term(arg_value)?);
                }
                "predicate" => {
                    predicate = Some(self.format_rdf_term(arg_value)?);
                }
                "object" => {
                    object = Some(self.format_rdf_term(arg_value)?);
                }
                _ => {}
            }
        }

        match (subject, predicate, object) {
            (Some(s), Some(p), Some(o)) => Ok(format!("{s} {p} {o} .")),
            _ => Err(anyhow!(
                "Missing required arguments: subject, predicate, or object"
            )),
        }
    }

    fn extract_triples_from_arguments(&self, arguments: &[(String, Value)]) -> Result<Vec<String>> {
        for (arg_name, arg_value) in arguments {
            if arg_name == "triples" {
                if let Value::ListValue(triple_list) = arg_value {
                    let mut result = Vec::new();
                    for triple_value in triple_list {
                        if let Value::ObjectValue(triple_obj) = triple_value {
                            let triple_args: Vec<(String, Value)> = triple_obj
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect();
                            result.push(self.extract_triple_from_arguments(&triple_args)?);
                        }
                    }
                    return Ok(result);
                }
            }
        }
        Err(anyhow!("No triples argument found"))
    }

    fn extract_old_triple_from_arguments(&self, arguments: &[(String, Value)]) -> Result<String> {
        for (arg_name, arg_value) in arguments {
            if arg_name == "old" {
                if let Value::ObjectValue(old_obj) = arg_value {
                    let old_args: Vec<(String, Value)> = old_obj
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    return self.extract_triple_from_arguments(&old_args);
                }
            }
        }
        Err(anyhow!("No old triple argument found"))
    }

    fn extract_new_triple_from_arguments(&self, arguments: &[(String, Value)]) -> Result<String> {
        for (arg_name, arg_value) in arguments {
            if arg_name == "new" {
                if let Value::ObjectValue(new_obj) = arg_value {
                    let new_args: Vec<(String, Value)> = new_obj
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    return self.extract_triple_from_arguments(&new_args);
                }
            }
        }
        Err(anyhow!("No new triple argument found"))
    }

    fn format_rdf_term(&self, value: &Value) -> Result<String> {
        match value {
            Value::StringValue(s) => {
                // Check if it's a URI (contains :// or starts with http)
                if s.contains("://") || s.starts_with("http") {
                    Ok(format!("<{s}>"))
                } else if s.starts_with("_:") {
                    // Blank node
                    Ok(s.clone())
                } else {
                    // Literal string
                    Ok(format!("\"{}\"", s.replace('\"', "\\\"")))
                }
            }
            Value::IntValue(i) => Ok(format!(
                "\"{i}\"^^<http://www.w3.org/2001/XMLSchema#integer>"
            )),
            Value::FloatValue(f) => Ok(format!(
                "\"{f}\"^^<http://www.w3.org/2001/XMLSchema#decimal>"
            )),
            Value::BooleanValue(b) => Ok(format!(
                "\"{b}\"^^<http://www.w3.org/2001/XMLSchema#boolean>"
            )),
            Value::ObjectValue(obj) => {
                // Handle typed literals
                if let (Some(Value::StringValue(value)), Some(Value::StringValue(datatype))) =
                    (obj.get("value"), obj.get("datatype"))
                {
                    Ok(format!("\"{value}\"^^<{datatype}>"))
                } else if let (Some(Value::StringValue(value)), Some(Value::StringValue(lang))) =
                    (obj.get("value"), obj.get("language"))
                {
                    Ok(format!("\"{value}\"@{lang}"))
                } else {
                    Err(anyhow!("Invalid RDF term object format"))
                }
            }
            _ => Err(anyhow!("Unsupported value type for RDF term")),
        }
    }

    fn translate_root_selection_set(
        &self,
        selection_set: &SelectionSet,
        builder: &mut SparqlQueryBuilder,
        context: &QueryContext,
    ) -> Result<()> {
        for selection in &selection_set.selections {
            match selection {
                Selection::Field(field) => {
                    self.translate_root_field(field, builder, context)?;
                }
                Selection::InlineFragment(fragment) => {
                    // Handle inline fragments
                    self.translate_root_selection_set(&fragment.selection_set, builder, context)?;
                }
                Selection::FragmentSpread(fragment_spread) => {
                    // Handle fragment spreads
                    if let Some(fragment_def) =
                        context.fragments.get(&fragment_spread.fragment_name)
                    {
                        self.translate_root_selection_set(
                            &fragment_def.selection_set,
                            builder,
                            context,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn translate_root_field(
        &self,
        field: &Field,
        builder: &mut SparqlQueryBuilder,
        context: &QueryContext,
    ) -> Result<()> {
        match field.name.as_str() {
            "hello" | "version" => {
                // These are computed fields, no SPARQL needed
            }
            "triples" => {
                builder.add_count_query("?s ?p ?o");
            }
            "subjects" => {
                let limit = self
                    .extract_limit_from_field(field)
                    .unwrap_or(self.config.default_limit);
                builder.add_distinct_select("?s");
                builder.add_where_pattern("?s ?p ?o");
                builder.set_limit(limit);
            }
            "predicates" => {
                let limit = self
                    .extract_limit_from_field(field)
                    .unwrap_or(self.config.default_limit);
                builder.add_distinct_select("?p");
                builder.add_where_pattern("?s ?p ?o");
                builder.set_limit(limit);
            }
            "objects" => {
                let limit = self
                    .extract_limit_from_field(field)
                    .unwrap_or(self.config.default_limit);
                builder.add_distinct_select("?o");
                builder.add_where_pattern("?s ?p ?o");
                builder.set_limit(limit);
            }
            "sparql" => {
                // Raw SPARQL - handled separately in resolvers
            }
            _ => {
                // Handle vocabulary-based fields
                if let Some(ref vocabulary) = self.vocabulary {
                    self.translate_vocabulary_field(field, builder, context, vocabulary)?;
                } else {
                    // Default pattern for unknown fields
                    let var_name = format!("?{}", field.name);
                    builder.add_select(&var_name);
                    builder.add_where_pattern(&format!(
                        "?s {} {}",
                        self.field_to_predicate(&field.name),
                        var_name
                    ));
                }
            }
        }
        Ok(())
    }

    fn translate_vocabulary_field(
        &self,
        field: &Field,
        builder: &mut SparqlQueryBuilder,
        context: &QueryContext,
        vocabulary: &RdfVocabulary,
    ) -> Result<()> {
        // Look for matching classes in the vocabulary
        let field_name_variations = vec![
            field.name.clone(),
            self.pluralize(&field.name),
            self.singularize(&field.name),
        ];

        for class_uri in vocabulary.classes.keys() {
            let class_name = self.uri_to_graphql_name(class_uri);
            let class_name_lower = class_name.to_lowercase();

            for variation in &field_name_variations {
                if variation.to_lowercase() == class_name_lower
                    || variation.to_lowercase() == self.pluralize(&class_name_lower)
                {
                    // This field corresponds to a vocabulary class
                    return self
                        .translate_class_query(field, class_uri, builder, context, vocabulary);
                }
            }
        }

        // No matching class found, treat as property
        self.translate_property_field(field, builder, context, vocabulary)
    }

    fn translate_class_query(
        &self,
        field: &Field,
        class_uri: &str,
        builder: &mut SparqlQueryBuilder,
        context: &QueryContext,
        vocabulary: &RdfVocabulary,
    ) -> Result<()> {
        let subject_var = "?subject";

        // Add type constraint
        builder.add_where_pattern(&format!("{subject_var} a <{class_uri}>"));

        // Handle arguments
        let limit = self
            .extract_limit_from_field(field)
            .unwrap_or(self.config.default_limit);
        let offset = self.extract_offset_from_field(field).unwrap_or(0);

        if let Some(where_filter) = self.extract_where_from_field(field) {
            builder.add_filter(&where_filter);
        }

        // Handle nested selections
        if let Some(ref selection_set) = field.selection_set {
            self.translate_object_selection_set(
                selection_set,
                subject_var,
                builder,
                context,
                vocabulary,
            )?;
        } else {
            // If no selection set, just return the URI
            builder.add_select(subject_var);
        }

        builder.set_limit(limit);
        if offset > 0 {
            builder.set_offset(offset);
        }

        Ok(())
    }

    fn translate_object_selection_set(
        &self,
        selection_set: &SelectionSet,
        subject_var: &str,
        builder: &mut SparqlQueryBuilder,
        context: &QueryContext,
        vocabulary: &RdfVocabulary,
    ) -> Result<()> {
        for selection in &selection_set.selections {
            match selection {
                Selection::Field(field) => {
                    self.translate_object_field(field, subject_var, builder, context, vocabulary)?;
                }
                Selection::InlineFragment(fragment) => {
                    self.translate_object_selection_set(
                        &fragment.selection_set,
                        subject_var,
                        builder,
                        context,
                        vocabulary,
                    )?;
                }
                Selection::FragmentSpread(fragment_spread) => {
                    if let Some(fragment_def) =
                        context.fragments.get(&fragment_spread.fragment_name)
                    {
                        self.translate_object_selection_set(
                            &fragment_def.selection_set,
                            subject_var,
                            builder,
                            context,
                            vocabulary,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn translate_object_field(
        &self,
        field: &Field,
        subject_var: &str,
        builder: &mut SparqlQueryBuilder,
        context: &QueryContext,
        vocabulary: &RdfVocabulary,
    ) -> Result<()> {
        match field.name.as_str() {
            "id" | "uri" => {
                builder.add_select(subject_var);
            }
            _ => {
                // Look for matching property
                let property_uri = self.find_property_uri(&field.name, vocabulary);
                if let Some(prop_uri) = property_uri {
                    let field_var = format!("?{}", field.name);
                    builder.add_select(&field_var);

                    if let Some(property) = vocabulary.properties.get(&prop_uri) {
                        match property.property_type {
                            PropertyType::ObjectProperty => {
                                builder.add_where_pattern(&format!(
                                    "{subject_var} <{prop_uri}> {field_var}"
                                ));

                                // Handle nested object selections
                                if let Some(ref selection_set) = field.selection_set {
                                    self.translate_object_selection_set(
                                        selection_set,
                                        &field_var,
                                        builder,
                                        context,
                                        vocabulary,
                                    )?;
                                }
                            }
                            PropertyType::DataProperty | PropertyType::AnnotationProperty => {
                                builder.add_where_pattern(&format!(
                                    "{subject_var} <{prop_uri}> {field_var}"
                                ));
                            }
                        }
                    } else {
                        // Fallback
                        builder
                            .add_where_pattern(&format!("{subject_var} <{prop_uri}> {field_var}"));
                    }
                } else {
                    // Fallback to predicate mapping
                    let predicate = self.field_to_predicate(&field.name);
                    let field_var = format!("?{}", field.name);
                    builder.add_select(&field_var);
                    builder.add_where_pattern(&format!("{subject_var} {predicate} {field_var}"));
                }
            }
        }
        Ok(())
    }

    fn translate_property_field(
        &self,
        field: &Field,
        builder: &mut SparqlQueryBuilder,
        _context: &QueryContext,
        _vocabulary: &RdfVocabulary,
    ) -> Result<()> {
        // Default property field translation
        let var_name = format!("?{}", field.name);
        let predicate = self.field_to_predicate(&field.name);

        builder.add_select(&var_name);
        builder.add_where_pattern(&format!("?s {predicate} {var_name}"));

        Ok(())
    }

    fn find_property_uri(&self, field_name: &str, vocabulary: &RdfVocabulary) -> Option<String> {
        // Try to find a property that matches the field name
        for (prop_uri, property) in &vocabulary.properties {
            if let Some(ref label) = property.label {
                if label.to_lowercase() == field_name.to_lowercase() {
                    return Some(prop_uri.clone());
                }
            }

            // Also try matching the local name from the URI
            let local_name = prop_uri.split(['#', '/']).next_back().unwrap_or("");
            if local_name.to_lowercase() == field_name.to_lowercase() {
                return Some(prop_uri.clone());
            }
        }
        None
    }

    fn uri_to_graphql_name(&self, uri: &str) -> String {
        if let Some(fragment) = uri.split('#').next_back() {
            self.to_pascal_case(fragment)
        } else if let Some(segment) = uri.split('/').next_back() {
            self.to_pascal_case(segment)
        } else {
            "Resource".to_string()
        }
    }

    fn to_pascal_case(&self, input: &str) -> String {
        let mut result = String::new();
        let mut capitalize_next = true;

        for ch in input.chars() {
            if ch.is_alphanumeric() {
                if capitalize_next {
                    result.push(ch.to_uppercase().next().unwrap_or(ch));
                    capitalize_next = false;
                } else {
                    result.push(ch);
                }
            } else {
                capitalize_next = true;
            }
        }

        result
    }

    fn pluralize(&self, word: &str) -> String {
        if word.ends_with('s') || word.ends_with("sh") || word.ends_with("ch") {
            format!("{word}es")
        } else if let Some(stripped) = word.strip_suffix('y') {
            format!("{stripped}ies")
        } else {
            format!("{word}s")
        }
    }

    fn singularize(&self, word: &str) -> String {
        if let Some(stripped) = word.strip_suffix("ies") {
            format!("{stripped}y")
        } else if word.ends_with("es") && (word.ends_with("shes") || word.ends_with("ches")) {
            word[..word.len() - 2].to_string()
        } else if word.ends_with('s') && word.len() > 1 {
            word[..word.len() - 1].to_string()
        } else {
            word.to_string()
        }
    }

    #[allow(dead_code)]
    fn build_sparql_from_selection_set(
        &self,
        selection_set: &SelectionSet,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let mut select_vars = Vec::new();
        let mut where_patterns = Vec::new();

        for selection in &selection_set.selections {
            match selection {
                Selection::Field(field) => {
                    match field.name.as_str() {
                        "hello" | "version" => {
                            // These are computed fields, no SPARQL needed
                        }
                        "triples" => {
                            // Count query
                            select_vars.push("(COUNT(*) as ?count)".to_string());
                            where_patterns.push("?s ?p ?o".to_string());
                        }
                        "subjects" => {
                            select_vars.push("DISTINCT ?s".to_string());
                            where_patterns.push("?s ?p ?o".to_string());

                            // Handle limit argument
                            if let Some(_limit) = self.extract_limit_from_field(field) {
                                // Limit will be added later
                            }
                        }
                        "predicates" => {
                            select_vars.push("DISTINCT ?p".to_string());
                            where_patterns.push("?s ?p ?o".to_string());
                        }
                        "objects" => {
                            select_vars.push("DISTINCT ?o".to_string());
                            where_patterns.push("?s ?p ?o".to_string());
                        }
                        "sparql" => {
                            // Raw SPARQL - handled separately
                        }
                        _ => {
                            // Default pattern for other fields
                            let var_name = format!("?{}", field.name);
                            select_vars.push(var_name.clone());
                            where_patterns.push(format!(
                                "?s {} {}",
                                self.field_to_predicate(&field.name),
                                var_name
                            ));
                        }
                    }
                }
                _ => {
                    // Handle fragments if needed
                }
            }
        }

        Ok((select_vars, where_patterns))
    }

    fn extract_limit_from_field(&self, field: &Field) -> Option<usize> {
        for arg in &field.arguments {
            if arg.name == "limit" {
                if let Value::IntValue(i) = &arg.value {
                    let limit = (*i as usize).min(self.config.max_limit);
                    return Some(limit);
                }
            }
        }
        None
    }

    fn extract_offset_from_field(&self, field: &Field) -> Option<usize> {
        for arg in &field.arguments {
            if arg.name == "offset" {
                if let Value::IntValue(i) = &arg.value {
                    return Some(*i as usize);
                }
            }
        }
        None
    }

    fn extract_where_from_field(&self, field: &Field) -> Option<String> {
        for arg in &field.arguments {
            if arg.name == "where" {
                if let Value::StringValue(s) = &arg.value {
                    return Some(s.clone());
                }
            }
        }
        None
    }

    fn field_to_predicate(&self, field_name: &str) -> String {
        // Simple mapping - in a real implementation, this would use vocabulary mappings
        match field_name {
            "name" => "foaf:name".to_string(),
            "email" => "foaf:mbox".to_string(),
            "knows" => "foaf:knows".to_string(),
            "label" => "rdfs:label".to_string(),
            "comment" => "rdfs:comment".to_string(),
            _ => format!(":{field_name}"), // Default to local namespace
        }
    }

    /// Create a GraphQL schema from RDF vocabulary
    pub fn rdf_to_graphql_schema(&self, vocabulary_uri: &str) -> Result<String> {
        // Placeholder implementation
        Ok(format!(
            r#"
type Query {{
  hello: String
  version: String
  triples: Int
  subjects(limit: Int = 10): [String!]!
  predicates(limit: Int = 10): [String!]!
  objects(limit: Int = 10): [String!]!
  sparql(query: String!): String
}}

# Generated from vocabulary: {vocabulary_uri}
"#
        ))
    }
}

/// SPARQL query builder for constructing complex queries
struct SparqlQueryBuilder {
    prefixes: HashMap<String, String>,
    select_vars: Vec<String>,
    where_patterns: Vec<String>,
    filters: Vec<String>,
    group_by: Vec<String>,
    order_by: Vec<String>,
    limit: Option<usize>,
    offset: Option<usize>,
    distinct: bool,
    config: TranslationConfig,
    // UPDATE operation fields
    insert_data: Vec<String>,
    delete_data: Vec<String>,
    insert_where: Vec<String>,
    delete_where: Vec<String>,
    is_update_query: bool,
}

impl SparqlQueryBuilder {
    fn new(prefixes: &HashMap<String, String>, config: &TranslationConfig) -> Self {
        Self {
            prefixes: prefixes.clone(),
            select_vars: Vec::new(),
            where_patterns: Vec::new(),
            filters: Vec::new(),
            group_by: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
            distinct: false,
            config: config.clone(),
            insert_data: Vec::new(),
            delete_data: Vec::new(),
            insert_where: Vec::new(),
            delete_where: Vec::new(),
            is_update_query: false,
        }
    }

    fn add_select(&mut self, var: &str) {
        if !self.select_vars.contains(&var.to_string()) {
            self.select_vars.push(var.to_string());
        }
    }

    fn add_distinct_select(&mut self, var: &str) {
        self.distinct = true;
        self.add_select(var);
    }

    fn add_count_query(&mut self, pattern: &str) {
        self.select_vars.push("(COUNT(*) as ?count)".to_string());
        self.add_where_pattern(pattern);
    }

    fn add_where_pattern(&mut self, pattern: &str) {
        self.where_patterns.push(pattern.to_string());
    }

    fn add_filter(&mut self, filter: &str) {
        self.filters.push(filter.to_string());
    }

    fn set_limit(&mut self, limit: usize) {
        self.limit = Some(limit.min(self.config.max_limit));
    }

    fn set_offset(&mut self, offset: usize) {
        self.offset = Some(offset);
    }

    fn add_insert_data(&mut self, triple: &str) {
        self.is_update_query = true;
        self.insert_data.push(triple.to_string());
    }

    fn add_delete_data(&mut self, triple: &str) {
        self.is_update_query = true;
        self.delete_data.push(triple.to_string());
    }

    #[allow(dead_code)]
    fn add_insert_where(&mut self, triple: &str, condition: &str) {
        self.is_update_query = true;
        self.insert_where
            .push(format!("{triple} WHERE {{ {condition} }}"));
    }

    #[allow(dead_code)]
    fn add_delete_where(&mut self, triple: &str, condition: &str) {
        self.is_update_query = true;
        self.delete_where
            .push(format!("{triple} WHERE {{ {condition} }}"));
    }

    fn build(&self) -> Result<String> {
        let mut query = String::new();

        // Add prefixes
        for (prefix, uri) in &self.prefixes {
            writeln!(query, "PREFIX {prefix}: <{uri}>")?;
        }

        if !query.is_empty() {
            writeln!(query)?;
        }

        if self.is_update_query {
            // Build UPDATE query
            self.build_update_query(&mut query)?;
        } else {
            // Build SELECT query
            self.build_select_query(&mut query)?;
        }

        Ok(query)
    }

    fn build_select_query(&self, query: &mut String) -> Result<()> {
        // Add SELECT clause
        if self.select_vars.is_empty() {
            write!(query, "SELECT *")?;
        } else {
            let distinct_clause = if self.distinct { "DISTINCT " } else { "" };
            write!(
                query,
                "SELECT {}{}",
                distinct_clause,
                self.select_vars.join(" ")
            )?;
        }

        // Add WHERE clause
        if !self.where_patterns.is_empty() {
            writeln!(query)?;
            writeln!(query, "WHERE {{")?;
            for pattern in &self.where_patterns {
                writeln!(query, "  {pattern}")?;
            }

            // Add filters
            for filter in &self.filters {
                writeln!(query, "  FILTER({filter})")?;
            }

            write!(query, "}}")?
        }

        // Add GROUP BY
        if !self.group_by.is_empty() {
            writeln!(query)?;
            write!(query, "GROUP BY {}", self.group_by.join(" "))?;
        }

        // Add ORDER BY
        if !self.order_by.is_empty() {
            writeln!(query)?;
            write!(query, "ORDER BY {}", self.order_by.join(" "))?;
        }

        // Add LIMIT and OFFSET
        if let Some(limit) = self.limit {
            writeln!(query)?;
            write!(query, "LIMIT {limit}")?;
        }

        if let Some(offset) = self.offset {
            writeln!(query)?;
            write!(query, "OFFSET {offset}")?;
        }

        Ok(())
    }

    fn build_update_query(&self, query: &mut String) -> Result<()> {
        let mut operations = Vec::new();

        // Add DELETE DATA operations
        if !self.delete_data.is_empty() {
            let mut delete_block = String::new();
            writeln!(delete_block, "DELETE DATA {{")?;
            for triple in &self.delete_data {
                writeln!(delete_block, "  {triple}")?;
            }
            write!(delete_block, "}}")?;
            operations.push(delete_block);
        }

        // Add INSERT DATA operations
        if !self.insert_data.is_empty() {
            let mut insert_block = String::new();
            writeln!(insert_block, "INSERT DATA {{")?;
            for triple in &self.insert_data {
                writeln!(insert_block, "  {triple}")?;
            }
            write!(insert_block, "}}")?;
            operations.push(insert_block);
        }

        // Add DELETE WHERE operations
        for delete_where in &self.delete_where {
            operations.push(format!("DELETE {delete_where}"));
        }

        // Add INSERT WHERE operations
        for insert_where in &self.insert_where {
            operations.push(format!("INSERT {insert_where}"));
        }

        // Join operations with semicolons
        write!(query, "{}", operations.join(";\n"))?;

        Ok(())
    }

    /// Add group by clause
    #[allow(dead_code)]
    fn add_group_by(&mut self, group: &str) {
        self.group_by.push(group.to_string());
    }

    /// Add order by clause
    #[allow(dead_code)]
    fn add_order_by(&mut self, order: &str) {
        self.order_by.push(order.to_string());
    }
}

impl Default for RdfGraphQLMapper {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for QueryContext {
    fn default() -> Self {
        Self::new()
    }
}
