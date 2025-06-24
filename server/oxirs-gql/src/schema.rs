//! GraphQL schema generation from RDF ontologies

use crate::types::*;
use crate::rdf_scalars::RdfScalars;
use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

/// RDF vocabulary information extracted from an ontology
#[derive(Debug, Clone)]
pub struct RdfVocabulary {
    pub classes: HashMap<String, RdfClass>,
    pub properties: HashMap<String, RdfProperty>,
    pub namespaces: HashMap<String, String>,
}

/// RDF class information
#[derive(Debug, Clone)]
pub struct RdfClass {
    pub uri: String,
    pub label: Option<String>,
    pub comment: Option<String>,
    pub super_classes: Vec<String>,
    pub properties: Vec<String>,
}

/// RDF property information
#[derive(Debug, Clone)]
pub struct RdfProperty {
    pub uri: String,
    pub label: Option<String>,
    pub comment: Option<String>,
    pub domain: Vec<String>,
    pub range: Vec<String>,
    pub property_type: PropertyType,
    pub functional: bool,
    pub inverse_functional: bool,
}

/// Type of RDF property
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyType {
    DataProperty,
    ObjectProperty,
    AnnotationProperty,
}

/// Configuration for schema generation
#[derive(Debug, Clone)]
pub struct SchemaGenerationConfig {
    pub include_deprecated: bool,
    pub max_depth: usize,
    pub custom_scalars: HashMap<String, String>,
    pub type_mappings: HashMap<String, String>,
    pub exclude_classes: HashSet<String>,
    pub exclude_properties: HashSet<String>,
    pub enable_introspection: bool,
    pub enable_mutations: bool,
    pub enable_subscriptions: bool,
}

impl Default for SchemaGenerationConfig {
    fn default() -> Self {
        Self {
            include_deprecated: false,
            max_depth: 10,
            custom_scalars: HashMap::new(),
            type_mappings: Self::default_type_mappings(),
            exclude_classes: HashSet::new(),
            exclude_properties: HashSet::new(),
            enable_introspection: true,
            enable_mutations: false,
            enable_subscriptions: false,
        }
    }
}

impl SchemaGenerationConfig {
    fn default_type_mappings() -> HashMap<String, String> {
        let mut mappings = HashMap::new();
        
        // XSD mappings
        mappings.insert("http://www.w3.org/2001/XMLSchema#string".to_string(), "String".to_string());
        mappings.insert("http://www.w3.org/2001/XMLSchema#int".to_string(), "Int".to_string());
        mappings.insert("http://www.w3.org/2001/XMLSchema#integer".to_string(), "Int".to_string());
        mappings.insert("http://www.w3.org/2001/XMLSchema#long".to_string(), "Int".to_string());
        mappings.insert("http://www.w3.org/2001/XMLSchema#float".to_string(), "Float".to_string());
        mappings.insert("http://www.w3.org/2001/XMLSchema#double".to_string(), "Float".to_string());
        mappings.insert("http://www.w3.org/2001/XMLSchema#decimal".to_string(), "Float".to_string());
        mappings.insert("http://www.w3.org/2001/XMLSchema#boolean".to_string(), "Boolean".to_string());
        mappings.insert("http://www.w3.org/2001/XMLSchema#dateTime".to_string(), "DateTime".to_string());
        mappings.insert("http://www.w3.org/2001/XMLSchema#date".to_string(), "DateTime".to_string());
        mappings.insert("http://www.w3.org/2001/XMLSchema#time".to_string(), "DateTime".to_string());
        mappings.insert("http://www.w3.org/2001/XMLSchema#duration".to_string(), "Duration".to_string());
        mappings.insert("http://www.w3.org/2001/XMLSchema#anyURI".to_string(), "IRI".to_string());
        
        // RDF mappings
        mappings.insert("http://www.w3.org/1999/02/22-rdf-syntax-ns#langString".to_string(), "LangString".to_string());
        mappings.insert("http://www.w3.org/2000/01/rdf-schema#Literal".to_string(), "Literal".to_string());
        mappings.insert("http://www.w3.org/1999/02/22-rdf-syntax-ns#PlainLiteral".to_string(), "String".to_string());
        
        // RDFS mappings
        mappings.insert("http://www.w3.org/2000/01/rdf-schema#Resource".to_string(), "IRI".to_string());
        
        mappings
    }
}

/// Schema generator that converts RDF ontologies to GraphQL schemas
pub struct SchemaGenerator {
    config: SchemaGenerationConfig,
    vocabulary: Option<RdfVocabulary>,
}

impl SchemaGenerator {
    pub fn new() -> Self {
        Self {
            config: SchemaGenerationConfig::default(),
            vocabulary: None,
        }
    }
    
    pub fn with_config(mut self, config: SchemaGenerationConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Set the RDF vocabulary to generate schema from
    pub fn with_vocabulary(mut self, vocabulary: RdfVocabulary) -> Self {
        self.vocabulary = Some(vocabulary);
        self
    }
    
    /// Generate GraphQL schema from loaded RDF vocabulary
    pub fn generate_schema(&self) -> Result<Schema> {
        let vocabulary = self.vocabulary.as_ref()
            .ok_or_else(|| anyhow!("No vocabulary loaded"))?;
        
        let mut schema = Schema::new();
        
        // Add RDF-specific scalar types
        schema.add_type(GraphQLType::Scalar(RdfScalars::iri()));
        schema.add_type(GraphQLType::Scalar(RdfScalars::literal()));
        schema.add_type(GraphQLType::Scalar(RdfScalars::datetime()));
        schema.add_type(GraphQLType::Scalar(RdfScalars::duration()));
        schema.add_type(GraphQLType::Scalar(RdfScalars::geolocation()));
        schema.add_type(GraphQLType::Scalar(RdfScalars::lang_string()));
        
        // Generate object types from RDF classes
        for (class_uri, rdf_class) in &vocabulary.classes {
            if self.config.exclude_classes.contains(class_uri) {
                continue;
            }
            
            let object_type = self.generate_object_type_from_class(rdf_class, vocabulary)?;
            schema.add_type(GraphQLType::Object(object_type));
        }
        
        // Generate Query type
        let query_type = self.generate_query_type(vocabulary)?;
        schema.add_type(GraphQLType::Object(query_type));
        schema.set_query_type("Query".to_string());
        
        // Generate Mutation type if enabled
        if self.config.enable_mutations {
            let mutation_type = self.generate_mutation_type(vocabulary)?;
            schema.add_type(GraphQLType::Object(mutation_type));
            schema.set_mutation_type("Mutation".to_string());
        }
        
        // Generate Subscription type if enabled
        if self.config.enable_subscriptions {
            let subscription_type = self.generate_subscription_type(vocabulary)?;
            schema.add_type(GraphQLType::Object(subscription_type));
            schema.set_subscription_type("Subscription".to_string());
        }
        
        Ok(schema)
    }
    
    /// Generate GraphQL schema SDL from RDF ontology
    pub fn generate_from_ontology(&self, ontology_uri: &str) -> Result<String> {
        // For now, load a mock vocabulary - TODO: implement real ontology parsing
        let vocabulary = self.load_mock_vocabulary(ontology_uri)?;
        
        let schema_with_vocab = Self::new()
            .with_config(self.config.clone())
            .with_vocabulary(vocabulary);
        
        let schema = schema_with_vocab.generate_schema()?;
        Ok(self.schema_to_sdl(&schema))
    }
    
    /// Generate GraphQL schema from RDF store containing ontology data
    pub fn generate_from_store(&self, store: &crate::RdfStore) -> Result<String> {
        let vocabulary = self.extract_vocabulary_from_store(store)?;
        
        let schema_with_vocab = Self::new()
            .with_config(self.config.clone())
            .with_vocabulary(vocabulary);
        
        let schema = schema_with_vocab.generate_schema()?;
        Ok(self.schema_to_sdl(&schema))
    }
    
    /// Extract RDF vocabulary from a store using SPARQL queries
    pub fn extract_vocabulary_from_store(&self, store: &crate::RdfStore) -> Result<RdfVocabulary> {
        let mut classes = HashMap::new();
        let mut properties = HashMap::new();
        let mut namespaces = HashMap::new();
        
        // Extract namespaces
        namespaces.insert("rdf".to_string(), "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string());
        namespaces.insert("rdfs".to_string(), "http://www.w3.org/2000/01/rdf-schema#".to_string());
        namespaces.insert("owl".to_string(), "http://www.w3.org/2002/07/owl#".to_string());
        
        // Extract classes using SPARQL
        let class_query = r#"
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            
            SELECT DISTINCT ?class ?label ?comment ?superClass
            WHERE {
                {
                    ?class a rdfs:Class .
                } UNION {
                    ?class a owl:Class .
                }
                OPTIONAL { ?class rdfs:label ?label }
                OPTIONAL { ?class rdfs:comment ?comment }
                OPTIONAL { ?class rdfs:subClassOf ?superClass }
                FILTER(!isBlank(?class))
            }
        "#;
        
        if let Ok(results) = store.query(class_query) {
            self.process_class_results(results, &mut classes)?;
        }
        
        // Extract properties using SPARQL
        let property_query = r#"
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            
            SELECT DISTINCT ?property ?label ?comment ?domain ?range ?type
            WHERE {
                {
                    ?property a rdf:Property .
                    BIND("AnnotationProperty" as ?type)
                } UNION {
                    ?property a rdfs:Property .
                    BIND("DataProperty" as ?type)
                } UNION {
                    ?property a owl:DatatypeProperty .
                    BIND("DataProperty" as ?type)
                } UNION {
                    ?property a owl:ObjectProperty .
                    BIND("ObjectProperty" as ?type)
                } UNION {
                    ?property a owl:AnnotationProperty .
                    BIND("AnnotationProperty" as ?type)
                }
                OPTIONAL { ?property rdfs:label ?label }
                OPTIONAL { ?property rdfs:comment ?comment }
                OPTIONAL { ?property rdfs:domain ?domain }
                OPTIONAL { ?property rdfs:range ?range }
                FILTER(!isBlank(?property))
            }
        "#;
        
        if let Ok(results) = store.query(property_query) {
            self.process_property_results(results, &mut properties)?;
        }
        
        // Link properties to classes
        self.link_properties_to_classes(&mut classes, &properties);
        
        Ok(RdfVocabulary {
            classes,
            properties,
            namespaces,
        })
    }
    
    fn process_class_results(&self, results: oxigraph::sparql::QueryResults, classes: &mut HashMap<String, RdfClass>) -> Result<()> {
        use oxigraph::sparql::QueryResults;
        
        if let QueryResults::Solutions(solutions) = results {
            for solution in solutions {
                let solution = solution?;
                
                if let Some(class_term) = solution.get("class") {
                    let class_uri = class_term.to_string();
                    
                    let label = solution.get("label")
                        .map(|t| t.to_string())
                        .and_then(|s| self.extract_literal_value(&s));
                    
                    let comment = solution.get("comment")
                        .map(|t| t.to_string())
                        .and_then(|s| self.extract_literal_value(&s));
                    
                    let super_class = solution.get("superClass")
                        .map(|t| t.to_string());
                    
                    // Get or create class entry
                    let rdf_class = classes.entry(class_uri.clone()).or_insert_with(|| RdfClass {
                        uri: class_uri.clone(),
                        label: None,
                        comment: None,
                        super_classes: Vec::new(),
                        properties: Vec::new(),
                    });
                    
                    // Update class information
                    if label.is_some() {
                        rdf_class.label = label;
                    }
                    if comment.is_some() {
                        rdf_class.comment = comment;
                    }
                    if let Some(sc) = super_class {
                        if !rdf_class.super_classes.contains(&sc) {
                            rdf_class.super_classes.push(sc);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn process_property_results(&self, results: oxigraph::sparql::QueryResults, properties: &mut HashMap<String, RdfProperty>) -> Result<()> {
        use oxigraph::sparql::QueryResults;
        
        if let QueryResults::Solutions(solutions) = results {
            for solution in solutions {
                let solution = solution?;
                
                if let Some(property_term) = solution.get("property") {
                    let property_uri = property_term.to_string();
                    
                    let label = solution.get("label")
                        .map(|t| t.to_string())
                        .and_then(|s| self.extract_literal_value(&s));
                    
                    let comment = solution.get("comment")
                        .map(|t| t.to_string())
                        .and_then(|s| self.extract_literal_value(&s));
                    
                    let domain = solution.get("domain")
                        .map(|t| t.to_string());
                    
                    let range = solution.get("range")
                        .map(|t| t.to_string());
                    
                    let property_type = solution.get("type")
                        .map(|t| t.to_string())
                        .and_then(|s| self.extract_literal_value(&s))
                        .unwrap_or_else(|| "AnnotationProperty".to_string());
                    
                    let prop_type = match property_type.as_str() {
                        "DataProperty" => PropertyType::DataProperty,
                        "ObjectProperty" => PropertyType::ObjectProperty,
                        _ => PropertyType::AnnotationProperty,
                    };
                    
                    // Get or create property entry
                    let rdf_property = properties.entry(property_uri.clone()).or_insert_with(|| RdfProperty {
                        uri: property_uri.clone(),
                        label: None,
                        comment: None,
                        domain: Vec::new(),
                        range: Vec::new(),
                        property_type: prop_type,
                        functional: false,
                        inverse_functional: false,
                    });
                    
                    // Update property information
                    if label.is_some() {
                        rdf_property.label = label;
                    }
                    if comment.is_some() {
                        rdf_property.comment = comment;
                    }
                    if let Some(d) = domain {
                        if !rdf_property.domain.contains(&d) {
                            rdf_property.domain.push(d);
                        }
                    }
                    if let Some(r) = range {
                        if !rdf_property.range.contains(&r) {
                            rdf_property.range.push(r);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn extract_literal_value(&self, term_str: &str) -> Option<String> {
        // Extract literal value from RDF term string format
        if term_str.starts_with('"') {
            if let Some(end_quote) = term_str[1..].find('"') {
                return Some(term_str[1..end_quote + 1].to_string());
            }
        }
        None
    }
    
    fn link_properties_to_classes(&self, classes: &mut HashMap<String, RdfClass>, properties: &HashMap<String, RdfProperty>) {
        for (property_uri, property) in properties {
            for domain_class in &property.domain {
                if let Some(class) = classes.get_mut(domain_class) {
                    if !class.properties.contains(property_uri) {
                        class.properties.push(property_uri.clone());
                    }
                }
            }
        }
    }
    
    fn generate_object_type_from_class(&self, rdf_class: &RdfClass, vocabulary: &RdfVocabulary) -> Result<ObjectType> {
        let type_name = self.uri_to_graphql_name(&rdf_class.uri);
        let mut object_type = ObjectType::new(type_name);
        
        if let Some(ref comment) = rdf_class.comment {
            object_type = object_type.with_description(comment.clone());
        }
        
        // Add ID field
        object_type = object_type.with_field(
            "id".to_string(),
            FieldType::new(
                "id".to_string(),
                GraphQLType::NonNull(Box::new(GraphQLType::Scalar(BuiltinScalars::id()))),
            ).with_description("The unique identifier of this resource".to_string()),
        );
        
        // Add URI field
        object_type = object_type.with_field(
            "uri".to_string(),
            FieldType::new(
                "uri".to_string(),
                GraphQLType::NonNull(Box::new(GraphQLType::Scalar(RdfScalars::iri()))),
            ).with_description("The IRI of this resource".to_string()),
        );
        
        // Add fields from properties
        for property_uri in &rdf_class.properties {
            if self.config.exclude_properties.contains(property_uri) {
                continue;
            }
            
            if let Some(property) = vocabulary.properties.get(property_uri) {
                let field = self.generate_field_from_property(property, vocabulary)?;
                let field_name = self.uri_to_graphql_name(&property.uri);
                object_type = object_type.with_field(field_name, field);
            }
        }
        
        Ok(object_type)
    }
    
    fn generate_field_from_property(&self, property: &RdfProperty, vocabulary: &RdfVocabulary) -> Result<FieldType> {
        let field_name = self.uri_to_graphql_name(&property.uri);
        
        let field_type = match property.property_type {
            PropertyType::DataProperty => {
                self.generate_scalar_type_from_range(&property.range)?
            }
            PropertyType::ObjectProperty => {
                self.generate_object_type_from_range(&property.range, vocabulary)?
            }
            PropertyType::AnnotationProperty => {
                GraphQLType::Scalar(BuiltinScalars::string())
            }
        };
        
        // Make non-functional properties lists
        let final_type = if property.functional {
            field_type
        } else {
            GraphQLType::List(Box::new(field_type))
        };
        
        let mut field = FieldType::new(field_name, final_type);
        
        if let Some(ref comment) = property.comment {
            field = field.with_description(comment.clone());
        }
        
        // Add filter arguments for object properties
        if matches!(property.property_type, PropertyType::ObjectProperty) {
            field = field.with_argument(
                "where".to_string(),
                ArgumentType::new(
                    "where".to_string(),
                    GraphQLType::Scalar(BuiltinScalars::string()),
                ).with_description("SPARQL filter condition".to_string()),
            );
            
            field = field.with_argument(
                "limit".to_string(),
                ArgumentType::new(
                    "limit".to_string(),
                    GraphQLType::Scalar(BuiltinScalars::int()),
                ).with_default_value(crate::ast::Value::IntValue(10))
                .with_description("Maximum number of results".to_string()),
            );
            
            field = field.with_argument(
                "offset".to_string(),
                ArgumentType::new(
                    "offset".to_string(),
                    GraphQLType::Scalar(BuiltinScalars::int()),
                ).with_default_value(crate::ast::Value::IntValue(0))
                .with_description("Number of results to skip".to_string()),
            );
        }
        
        Ok(field)
    }
    
    fn generate_scalar_type_from_range(&self, range: &[String]) -> Result<GraphQLType> {
        if range.is_empty() {
            return Ok(GraphQLType::Scalar(BuiltinScalars::string()));
        }
        
        for range_uri in range {
            if let Some(mapped_type) = self.config.type_mappings.get(range_uri) {
                return match mapped_type.as_str() {
                    "String" => Ok(GraphQLType::Scalar(BuiltinScalars::string())),
                    "Int" => Ok(GraphQLType::Scalar(BuiltinScalars::int())),
                    "Float" => Ok(GraphQLType::Scalar(BuiltinScalars::float())),
                    "Boolean" => Ok(GraphQLType::Scalar(BuiltinScalars::boolean())),
                    "ID" => Ok(GraphQLType::Scalar(BuiltinScalars::id())),
                    "IRI" => Ok(GraphQLType::Scalar(RdfScalars::iri())),
                    "Literal" => Ok(GraphQLType::Scalar(RdfScalars::literal())),
                    "DateTime" => Ok(GraphQLType::Scalar(RdfScalars::datetime())),
                    "Duration" => Ok(GraphQLType::Scalar(RdfScalars::duration())),
                    "GeoLocation" => Ok(GraphQLType::Scalar(RdfScalars::geolocation())),
                    "LangString" => Ok(GraphQLType::Scalar(RdfScalars::lang_string())),
                    _ => Ok(GraphQLType::Scalar(BuiltinScalars::string())),
                };
            }
        }
        
        // Default to Literal for unknown ranges
        Ok(GraphQLType::Scalar(RdfScalars::literal()))
    }
    
    fn generate_object_type_from_range(&self, range: &[String], vocabulary: &RdfVocabulary) -> Result<GraphQLType> {
        if range.is_empty() {
            return Ok(GraphQLType::Scalar(RdfScalars::iri()));
        }
        
        // For now, return the first valid class in range
        for range_uri in range {
            if vocabulary.classes.contains_key(range_uri) {
                let type_name = self.uri_to_graphql_name(range_uri);
                // We assume the type will be generated elsewhere
                return Ok(GraphQLType::Object(ObjectType::new(type_name)));
            }
        }
        
        // Default to IRI if no class found
        Ok(GraphQLType::Scalar(RdfScalars::iri()))
    }
    
    fn generate_query_type(&self, vocabulary: &RdfVocabulary) -> Result<ObjectType> {
        let mut query_type = ObjectType::new("Query".to_string())
            .with_description("The root query type for accessing RDF data".to_string());
        
        // Add root field for each class
        for (class_uri, rdf_class) in &vocabulary.classes {
            if self.config.exclude_classes.contains(class_uri) {
                continue;
            }
            
            let type_name = self.uri_to_graphql_name(&rdf_class.uri);
            let field_name = self.pluralize(&self.to_camel_case(&type_name));
            
            // Collection query
            query_type = query_type.with_field(
                field_name.clone(),
                FieldType::new(
                    field_name.clone(),
                    GraphQLType::List(Box::new(GraphQLType::Object(ObjectType::new(type_name.clone())))),
                ).with_description(format!("Query all instances of {}", type_name))
                .with_argument(
                    "where".to_string(),
                    ArgumentType::new(
                        "where".to_string(),
                        GraphQLType::Scalar(BuiltinScalars::string()),
                    ).with_description("SPARQL filter condition".to_string()),
                )
                .with_argument(
                    "limit".to_string(),
                    ArgumentType::new(
                        "limit".to_string(),
                        GraphQLType::Scalar(BuiltinScalars::int()),
                    ).with_default_value(crate::ast::Value::IntValue(10))
                    .with_description("Maximum number of results".to_string()),
                )
                .with_argument(
                    "offset".to_string(),
                    ArgumentType::new(
                        "offset".to_string(),
                        GraphQLType::Scalar(BuiltinScalars::int()),
                    ).with_default_value(crate::ast::Value::IntValue(0))
                    .with_description("Number of results to skip".to_string()),
                ),
            );
            
            // Single item query
            let singular_field = self.to_camel_case(&type_name);
            query_type = query_type.with_field(
                singular_field.clone(),
                FieldType::new(
                    singular_field.clone(),
                    GraphQLType::Object(ObjectType::new(type_name.clone())),
                ).with_description(format!("Query a single {} by ID", type_name))
                .with_argument(
                    "id".to_string(),
                    ArgumentType::new(
                        "id".to_string(),
                        GraphQLType::NonNull(Box::new(GraphQLType::Scalar(BuiltinScalars::id()))),
                    ).with_description("The ID of the resource".to_string()),
                ),
            );
        }
        
        // Add SPARQL query field
        query_type = query_type.with_field(
            "sparql".to_string(),
            FieldType::new(
                "sparql".to_string(),
                GraphQLType::Scalar(BuiltinScalars::string()),
            ).with_description("Execute a raw SPARQL query".to_string())
            .with_argument(
                "query".to_string(),
                ArgumentType::new(
                    "query".to_string(),
                    GraphQLType::NonNull(Box::new(GraphQLType::Scalar(BuiltinScalars::string()))),
                ).with_description("The SPARQL query to execute".to_string()),
            ),
        );
        
        Ok(query_type)
    }
    
    fn generate_mutation_type(&self, _vocabulary: &RdfVocabulary) -> Result<ObjectType> {
        let mutation_type = ObjectType::new("Mutation".to_string())
            .with_description("The root mutation type for modifying RDF data".to_string());
        
        // TODO: Add mutation fields based on vocabulary
        Ok(mutation_type)
    }
    
    fn generate_subscription_type(&self, _vocabulary: &RdfVocabulary) -> Result<ObjectType> {
        let subscription_type = ObjectType::new("Subscription".to_string())
            .with_description("The root subscription type for real-time RDF data updates".to_string());
        
        // TODO: Add subscription fields based on vocabulary
        Ok(subscription_type)
    }
    
    fn uri_to_graphql_name(&self, uri: &str) -> String {
        if let Some(fragment) = uri.split('#').last() {
            self.to_pascal_case(fragment)
        } else if let Some(segment) = uri.split('/').last() {
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
    
    fn to_camel_case(&self, input: &str) -> String {
        let pascal = self.to_pascal_case(input);
        if let Some(first_char) = pascal.chars().next() {
            first_char.to_lowercase().collect::<String>() + &pascal[first_char.len_utf8()..]
        } else {
            pascal
        }
    }
    
    fn pluralize(&self, word: &str) -> String {
        if word.ends_with('s') || word.ends_with("sh") || word.ends_with("ch") {
            format!("{}es", word)
        } else if word.ends_with('y') {
            format!("{}ies", &word[..word.len()-1])
        } else {
            format!("{}s", word)
        }
    }
    
    fn schema_to_sdl(&self, schema: &Schema) -> String {
        let mut sdl = String::new();
        
        // Write schema definition
        writeln!(sdl, "schema {{").unwrap();
        if let Some(ref query) = schema.query_type {
            writeln!(sdl, "  query: {}", query).unwrap();
        }
        if let Some(ref mutation) = schema.mutation_type {
            writeln!(sdl, "  mutation: {}", mutation).unwrap();
        }
        if let Some(ref subscription) = schema.subscription_type {
            writeln!(sdl, "  subscription: {}", subscription).unwrap();
        }
        writeln!(sdl, "}}").unwrap();
        writeln!(sdl).unwrap();
        
        // Write type definitions
        for (_, graphql_type) in &schema.types {
            match graphql_type {
                GraphQLType::Object(obj) => {
                    self.write_object_type_sdl(&mut sdl, obj);
                }
                GraphQLType::Scalar(scalar) => {
                    if !["String", "Int", "Float", "Boolean", "ID"].contains(&scalar.name.as_str()) {
                        self.write_scalar_type_sdl(&mut sdl, scalar);
                    }
                }
                GraphQLType::Enum(enum_type) => {
                    self.write_enum_type_sdl(&mut sdl, enum_type);
                }
                GraphQLType::Interface(interface) => {
                    self.write_interface_type_sdl(&mut sdl, interface);
                }
                GraphQLType::Union(union_type) => {
                    self.write_union_type_sdl(&mut sdl, union_type);
                }
                _ => {} // Skip other types
            }
        }
        
        sdl
    }
    
    fn write_object_type_sdl(&self, sdl: &mut String, obj: &ObjectType) {
        if let Some(ref description) = obj.description {
            writeln!(sdl, "\"\"\"\n{}\n\"\"\"", description).unwrap();
        }
        
        write!(sdl, "type {}", obj.name).unwrap();
        
        if !obj.interfaces.is_empty() {
            write!(sdl, " implements {}", obj.interfaces.join(" & ")).unwrap();
        }
        
        writeln!(sdl, " {{").unwrap();
        
        for (_, field) in &obj.fields {
            self.write_field_sdl(sdl, field);
        }
        
        writeln!(sdl, "}}").unwrap();
        writeln!(sdl).unwrap();
    }
    
    fn write_field_sdl(&self, sdl: &mut String, field: &FieldType) {
        if let Some(ref description) = field.description {
            writeln!(sdl, "  \"{}\"", description).unwrap();
        }
        
        write!(sdl, "  {}", field.name).unwrap();
        
        if !field.arguments.is_empty() {
            write!(sdl, "(").unwrap();
            let args: Vec<String> = field.arguments.values()
                .map(|arg| format!("{}: {}", arg.name, arg.argument_type))
                .collect();
            write!(sdl, "{}", args.join(", ")).unwrap();
            write!(sdl, ")").unwrap();
        }
        
        writeln!(sdl, ": {}", field.field_type).unwrap();
    }
    
    fn write_scalar_type_sdl(&self, sdl: &mut String, scalar: &ScalarType) {
        if let Some(ref description) = scalar.description {
            writeln!(sdl, "\"\"\"\n{}\n\"\"\"", description).unwrap();
        }
        writeln!(sdl, "scalar {}", scalar.name).unwrap();
        writeln!(sdl).unwrap();
    }
    
    fn write_enum_type_sdl(&self, sdl: &mut String, enum_type: &EnumType) {
        if let Some(ref description) = enum_type.description {
            writeln!(sdl, "\"\"\"\n{}\n\"\"\"", description).unwrap();
        }
        writeln!(sdl, "enum {} {{", enum_type.name).unwrap();
        
        for (_, value) in &enum_type.values {
            if let Some(ref description) = value.description {
                writeln!(sdl, "  \"{}\"", description).unwrap();
            }
            writeln!(sdl, "  {}", value.name).unwrap();
        }
        
        writeln!(sdl, "}}").unwrap();
        writeln!(sdl).unwrap();
    }
    
    fn write_interface_type_sdl(&self, sdl: &mut String, interface: &InterfaceType) {
        if let Some(ref description) = interface.description {
            writeln!(sdl, "\"\"\"\n{}\n\"\"\"", description).unwrap();
        }
        writeln!(sdl, "interface {} {{", interface.name).unwrap();
        
        for (_, field) in &interface.fields {
            self.write_field_sdl(sdl, field);
        }
        
        writeln!(sdl, "}}").unwrap();
        writeln!(sdl).unwrap();
    }
    
    fn write_union_type_sdl(&self, sdl: &mut String, union_type: &UnionType) {
        if let Some(ref description) = union_type.description {
            writeln!(sdl, "\"\"\"\n{}\n\"\"\"", description).unwrap();
        }
        writeln!(sdl, "union {} = {}", union_type.name, union_type.types.join(" | ")).unwrap();
        writeln!(sdl).unwrap();
    }
    
    fn load_mock_vocabulary(&self, ontology_uri: &str) -> Result<RdfVocabulary> {
        // Enhanced mock vocabulary for demonstration
        let mut classes = HashMap::new();
        let mut properties = HashMap::new();
        let mut namespaces = HashMap::new();
        
        // Common namespaces
        namespaces.insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());
        namespaces.insert("schema".to_string(), "http://schema.org/".to_string());
        namespaces.insert("dbo".to_string(), "http://dbpedia.org/ontology/".to_string());
        namespaces.insert("dc".to_string(), "http://purl.org/dc/elements/1.1/".to_string());
        
        // FOAF Agent (base class)
        classes.insert(
            "http://xmlns.com/foaf/0.1/Agent".to_string(),
            RdfClass {
                uri: "http://xmlns.com/foaf/0.1/Agent".to_string(),
                label: Some("Agent".to_string()),
                comment: Some("An agent (eg. person, group, software or physical artifact)".to_string()),
                super_classes: vec![],
                properties: vec![
                    "http://xmlns.com/foaf/0.1/name".to_string(),
                ],
            },
        );
        
        // FOAF Person class
        classes.insert(
            "http://xmlns.com/foaf/0.1/Person".to_string(),
            RdfClass {
                uri: "http://xmlns.com/foaf/0.1/Person".to_string(),
                label: Some("Person".to_string()),
                comment: Some("A person".to_string()),
                super_classes: vec!["http://xmlns.com/foaf/0.1/Agent".to_string()],
                properties: vec![
                    "http://xmlns.com/foaf/0.1/name".to_string(),
                    "http://xmlns.com/foaf/0.1/email".to_string(),
                    "http://xmlns.com/foaf/0.1/knows".to_string(),
                    "http://xmlns.com/foaf/0.1/age".to_string(),
                    "http://xmlns.com/foaf/0.1/homepage".to_string(),
                ],
            },
        );
        
        // FOAF Organization class
        classes.insert(
            "http://xmlns.com/foaf/0.1/Organization".to_string(),
            RdfClass {
                uri: "http://xmlns.com/foaf/0.1/Organization".to_string(),
                label: Some("Organization".to_string()),
                comment: Some("An organization".to_string()),
                super_classes: vec!["http://xmlns.com/foaf/0.1/Agent".to_string()],
                properties: vec![
                    "http://xmlns.com/foaf/0.1/name".to_string(),
                    "http://xmlns.com/foaf/0.1/homepage".to_string(),
                ],
            },
        );
        
        // Schema.org Product class
        classes.insert(
            "http://schema.org/Product".to_string(),
            RdfClass {
                uri: "http://schema.org/Product".to_string(),
                label: Some("Product".to_string()),
                comment: Some("Any offered product or service".to_string()),
                super_classes: vec![],
                properties: vec![
                    "http://schema.org/name".to_string(),
                    "http://schema.org/description".to_string(),
                    "http://schema.org/price".to_string(),
                    "http://schema.org/manufacturer".to_string(),
                ],
            },
        );
        
        // Properties
        let property_definitions = vec![
            ("http://xmlns.com/foaf/0.1/name", "name", "A name for some thing", PropertyType::DataProperty, vec!["http://xmlns.com/foaf/0.1/Agent"], vec!["http://www.w3.org/2001/XMLSchema#string"]),
            ("http://xmlns.com/foaf/0.1/email", "email", "An email address", PropertyType::DataProperty, vec!["http://xmlns.com/foaf/0.1/Person"], vec!["http://www.w3.org/2001/XMLSchema#string"]),
            ("http://xmlns.com/foaf/0.1/age", "age", "The age in years of some agent", PropertyType::DataProperty, vec!["http://xmlns.com/foaf/0.1/Person"], vec!["http://www.w3.org/2001/XMLSchema#int"]),
            ("http://xmlns.com/foaf/0.1/homepage", "homepage", "A homepage for some thing", PropertyType::DataProperty, vec!["http://xmlns.com/foaf/0.1/Agent"], vec!["http://www.w3.org/2001/XMLSchema#anyURI"]),
            ("http://xmlns.com/foaf/0.1/knows", "knows", "A person known by this person", PropertyType::ObjectProperty, vec!["http://xmlns.com/foaf/0.1/Person"], vec!["http://xmlns.com/foaf/0.1/Person"]),
            ("http://schema.org/name", "name", "The name of the item", PropertyType::DataProperty, vec!["http://schema.org/Product"], vec!["http://www.w3.org/2001/XMLSchema#string"]),
            ("http://schema.org/description", "description", "A description of the item", PropertyType::DataProperty, vec!["http://schema.org/Product"], vec!["http://www.w3.org/2001/XMLSchema#string"]),
            ("http://schema.org/price", "price", "The price of the product", PropertyType::DataProperty, vec!["http://schema.org/Product"], vec!["http://www.w3.org/2001/XMLSchema#decimal"]),
            ("http://schema.org/manufacturer", "manufacturer", "The manufacturer of the product", PropertyType::ObjectProperty, vec!["http://schema.org/Product"], vec!["http://xmlns.com/foaf/0.1/Organization"]),
        ];
        
        for (uri, label, comment, prop_type, domain, range) in property_definitions {
            properties.insert(
                uri.to_string(),
                RdfProperty {
                    uri: uri.to_string(),
                    label: Some(label.to_string()),
                    comment: Some(comment.to_string()),
                    domain: domain.into_iter().map(|s| s.to_string()).collect(),
                    range: range.into_iter().map(|s| s.to_string()).collect(),
                    property_type: prop_type,
                    functional: matches!(label, "email" | "age" | "homepage"),
                    inverse_functional: label == "email",
                },
            );
        }
        
        Ok(RdfVocabulary {
            classes,
            properties,
            namespaces,
        })
    }
}

impl Default for SchemaGenerator {
    fn default() -> Self {
        Self::new()
    }
}