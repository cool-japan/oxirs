//! GraphQL schema generator and RDF-to-GraphQL type mapping.
//!
//! Defines the [`SchemaGenerator`] together with the logic that turns an
//! [`RdfVocabulary`] into GraphQL object,
//! query, mutation and subscription types.

use crate::rdf_scalars::RdfScalars;
use crate::schema_types::{
    PropertyType, RdfClass, RdfProperty, RdfVocabulary, SchemaGenerationConfig,
};
use crate::types::*;
use anyhow::{anyhow, Result};

/// Schema generator that converts RDF ontologies to GraphQL schemas
pub struct SchemaGenerator {
    pub(crate) config: SchemaGenerationConfig,
    pub(crate) vocabulary: Option<RdfVocabulary>,
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
        let vocabulary = self
            .vocabulary
            .as_ref()
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
    pub async fn generate_from_ontology(&self, ontology_uri: &str) -> Result<String> {
        // Load and parse real RDF ontology from URI
        let vocabulary = self.load_ontology_from_uri(ontology_uri).await?;

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

    fn generate_object_type_from_class(
        &self,
        rdf_class: &RdfClass,
        vocabulary: &RdfVocabulary,
    ) -> Result<ObjectType> {
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
            )
            .with_description("The unique identifier of this resource".to_string()),
        );

        // Add URI field
        object_type = object_type.with_field(
            "uri".to_string(),
            FieldType::new(
                "uri".to_string(),
                GraphQLType::NonNull(Box::new(GraphQLType::Scalar(RdfScalars::iri()))),
            )
            .with_description("The IRI of this resource".to_string()),
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

    fn generate_field_from_property(
        &self,
        property: &RdfProperty,
        vocabulary: &RdfVocabulary,
    ) -> Result<FieldType> {
        let field_name = self.uri_to_graphql_name(&property.uri);

        let field_type = match property.property_type {
            PropertyType::DataProperty => self.generate_scalar_type_from_range(&property.range)?,
            PropertyType::ObjectProperty => {
                self.generate_object_type_from_range(&property.range, vocabulary)?
            }
            PropertyType::AnnotationProperty => GraphQLType::Scalar(BuiltinScalars::string()),
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
                )
                .with_description("SPARQL filter condition".to_string()),
            );

            field = field.with_argument(
                "limit".to_string(),
                ArgumentType::new(
                    "limit".to_string(),
                    GraphQLType::Scalar(BuiltinScalars::int()),
                )
                .with_default_value(crate::ast::Value::IntValue(10))
                .with_description("Maximum number of results".to_string()),
            );

            field = field.with_argument(
                "offset".to_string(),
                ArgumentType::new(
                    "offset".to_string(),
                    GraphQLType::Scalar(BuiltinScalars::int()),
                )
                .with_default_value(crate::ast::Value::IntValue(0))
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

    fn generate_object_type_from_range(
        &self,
        range: &[String],
        vocabulary: &RdfVocabulary,
    ) -> Result<GraphQLType> {
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
                    GraphQLType::List(Box::new(GraphQLType::Object(ObjectType::new(
                        type_name.clone(),
                    )))),
                )
                .with_description(format!("Query all instances of {type_name}"))
                .with_argument(
                    "where".to_string(),
                    ArgumentType::new(
                        "where".to_string(),
                        GraphQLType::Scalar(BuiltinScalars::string()),
                    )
                    .with_description("SPARQL filter condition".to_string()),
                )
                .with_argument(
                    "limit".to_string(),
                    ArgumentType::new(
                        "limit".to_string(),
                        GraphQLType::Scalar(BuiltinScalars::int()),
                    )
                    .with_default_value(crate::ast::Value::IntValue(10))
                    .with_description("Maximum number of results".to_string()),
                )
                .with_argument(
                    "offset".to_string(),
                    ArgumentType::new(
                        "offset".to_string(),
                        GraphQLType::Scalar(BuiltinScalars::int()),
                    )
                    .with_default_value(crate::ast::Value::IntValue(0))
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
                )
                .with_description(format!("Query a single {type_name} by ID"))
                .with_argument(
                    "id".to_string(),
                    ArgumentType::new(
                        "id".to_string(),
                        GraphQLType::NonNull(Box::new(GraphQLType::Scalar(BuiltinScalars::id()))),
                    )
                    .with_description("The ID of the resource".to_string()),
                ),
            );
        }

        // Add SPARQL query field
        query_type = query_type.with_field(
            "sparql".to_string(),
            FieldType::new(
                "sparql".to_string(),
                GraphQLType::Scalar(BuiltinScalars::string()),
            )
            .with_description("Execute a raw SPARQL query".to_string())
            .with_argument(
                "query".to_string(),
                ArgumentType::new(
                    "query".to_string(),
                    GraphQLType::NonNull(Box::new(GraphQLType::Scalar(BuiltinScalars::string()))),
                )
                .with_description("The SPARQL query to execute".to_string()),
            ),
        );

        Ok(query_type)
    }

    fn generate_mutation_type(&self, vocabulary: &RdfVocabulary) -> Result<ObjectType> {
        let mut mutation_type = ObjectType::new("Mutation".to_string())
            .with_description("The root mutation type for modifying RDF data".to_string());

        // Add CRUD operations for each class
        for (class_uri, rdf_class) in &vocabulary.classes {
            if self.config.exclude_classes.contains(class_uri) {
                continue;
            }

            let type_name = self.uri_to_graphql_name(&rdf_class.uri);
            let input_type_name = format!("{type_name}Input");
            let update_input_type_name = format!("{type_name}UpdateInput");

            // Create mutation for adding new instances
            mutation_type = mutation_type.with_field(
                format!("create{type_name}"),
                FieldType::new(
                    format!("create{type_name}"),
                    GraphQLType::Object(ObjectType::new(type_name.clone())),
                )
                .with_description(format!("Create a new {type_name}"))
                .with_argument(
                    "input".to_string(),
                    ArgumentType::new(
                        "input".to_string(),
                        GraphQLType::NonNull(Box::new(GraphQLType::InputObject(
                            InputObjectType::new(input_type_name.clone()),
                        ))),
                    )
                    .with_description(format!("Input data for creating a new {type_name}")),
                ),
            );

            // Update mutation
            mutation_type = mutation_type.with_field(
                format!("update{type_name}"),
                FieldType::new(
                    format!("update{type_name}"),
                    GraphQLType::Object(ObjectType::new(type_name.clone())),
                )
                .with_description(format!("Update an existing {type_name}"))
                .with_argument(
                    "id".to_string(),
                    ArgumentType::new(
                        "id".to_string(),
                        GraphQLType::NonNull(Box::new(GraphQLType::Scalar(BuiltinScalars::id()))),
                    )
                    .with_description("The ID of the resource to update".to_string()),
                )
                .with_argument(
                    "input".to_string(),
                    ArgumentType::new(
                        "input".to_string(),
                        GraphQLType::NonNull(Box::new(GraphQLType::InputObject(
                            InputObjectType::new(update_input_type_name),
                        ))),
                    )
                    .with_description(format!("Input data for updating the {type_name}")),
                ),
            );

            // Delete mutation
            mutation_type = mutation_type.with_field(
                format!("delete{type_name}"),
                FieldType::new(
                    format!("delete{type_name}"),
                    GraphQLType::Scalar(BuiltinScalars::boolean()),
                )
                .with_description(format!("Delete a {type_name}"))
                .with_argument(
                    "id".to_string(),
                    ArgumentType::new(
                        "id".to_string(),
                        GraphQLType::NonNull(Box::new(GraphQLType::Scalar(BuiltinScalars::id()))),
                    )
                    .with_description("The ID of the resource to delete".to_string()),
                ),
            );
        }

        // Add bulk operations
        mutation_type = mutation_type.with_field(
            "executeSparqlUpdate".to_string(),
            FieldType::new(
                "executeSparqlUpdate".to_string(),
                GraphQLType::Scalar(BuiltinScalars::boolean()),
            )
            .with_description("Execute a raw SPARQL UPDATE query".to_string())
            .with_argument(
                "update".to_string(),
                ArgumentType::new(
                    "update".to_string(),
                    GraphQLType::NonNull(Box::new(GraphQLType::Scalar(BuiltinScalars::string()))),
                )
                .with_description("The SPARQL UPDATE query to execute".to_string()),
            ),
        );

        // Add transaction support
        mutation_type = mutation_type.with_field(
            "executeTransaction".to_string(),
            FieldType::new(
                "executeTransaction".to_string(),
                GraphQLType::Scalar(BuiltinScalars::boolean()),
            )
            .with_description("Execute multiple SPARQL UPDATE queries in a transaction".to_string())
            .with_argument(
                "updates".to_string(),
                ArgumentType::new(
                    "updates".to_string(),
                    GraphQLType::NonNull(Box::new(GraphQLType::List(Box::new(
                        GraphQLType::Scalar(BuiltinScalars::string()),
                    )))),
                )
                .with_description("List of SPARQL UPDATE queries to execute".to_string()),
            ),
        );

        Ok(mutation_type)
    }

    fn generate_subscription_type(&self, vocabulary: &RdfVocabulary) -> Result<ObjectType> {
        let mut subscription_type = ObjectType::new("Subscription".to_string()).with_description(
            "The root subscription type for real-time RDF data updates".to_string(),
        );

        // Add change subscriptions for each class
        for (class_uri, rdf_class) in &vocabulary.classes {
            if self.config.exclude_classes.contains(class_uri) {
                continue;
            }

            let type_name = self.uri_to_graphql_name(&rdf_class.uri);
            let field_name = format!("{}Changed", self.to_camel_case(&type_name));

            // Resource change subscription
            subscription_type = subscription_type.with_field(
                field_name.clone(),
                FieldType::new(
                    field_name.clone(),
                    GraphQLType::Object(ObjectType::new(format!("{type_name}ChangeEvent"))),
                )
                .with_description(format!("Subscribe to changes for {type_name} instances"))
                .with_argument(
                    "id".to_string(),
                    ArgumentType::new("id".to_string(), GraphQLType::Scalar(BuiltinScalars::id()))
                        .with_description(
                            "Subscribe to changes for a specific resource ID".to_string(),
                        ),
                )
                .with_argument(
                    "changeType".to_string(),
                    ArgumentType::new(
                        "changeType".to_string(),
                        GraphQLType::Enum(EnumType::new("ChangeType".to_string())),
                    )
                    .with_description(
                        "Filter by change type (CREATED, UPDATED, DELETED)".to_string(),
                    ),
                ),
            );

            // Collection changes subscription
            let collection_field = format!("{}CollectionChanged", self.to_camel_case(&type_name));
            subscription_type = subscription_type.with_field(
                collection_field.clone(),
                FieldType::new(
                    collection_field.clone(),
                    GraphQLType::Object(ObjectType::new("CollectionChangeEvent".to_string())),
                )
                .with_description(format!("Subscribe to collection changes for {type_name}"))
                .with_argument(
                    "filter".to_string(),
                    ArgumentType::new(
                        "filter".to_string(),
                        GraphQLType::Scalar(BuiltinScalars::string()),
                    )
                    .with_description("SPARQL filter condition for subscription".to_string()),
                ),
            );
        }

        // Add property-specific change subscriptions
        for (property_uri, property) in &vocabulary.properties {
            if self.config.exclude_properties.contains(property_uri) {
                continue;
            }

            let property_name = self.uri_to_graphql_name(&property.uri);
            let field_name = format!("{}PropertyChanged", self.to_camel_case(&property_name));

            subscription_type = subscription_type.with_field(
                field_name.clone(),
                FieldType::new(
                    field_name.clone(),
                    GraphQLType::Object(ObjectType::new("PropertyChangeEvent".to_string())),
                )
                .with_description(format!(
                    "Subscribe to changes for the {property_name} property"
                ))
                .with_argument(
                    "subject".to_string(),
                    ArgumentType::new(
                        "subject".to_string(),
                        GraphQLType::Scalar(RdfScalars::iri()),
                    )
                    .with_description("The subject resource to monitor".to_string()),
                ),
            );
        }

        // Add query-based subscription
        subscription_type = subscription_type.with_field(
            "queryResultChanged".to_string(),
            FieldType::new(
                "queryResultChanged".to_string(),
                GraphQLType::Object(ObjectType::new("QueryResultChangeEvent".to_string())),
            )
            .with_description("Subscribe to changes in SPARQL query results".to_string())
            .with_argument(
                "query".to_string(),
                ArgumentType::new(
                    "query".to_string(),
                    GraphQLType::NonNull(Box::new(GraphQLType::Scalar(BuiltinScalars::string()))),
                )
                .with_description("The SPARQL query to monitor".to_string()),
            )
            .with_argument(
                "pollInterval".to_string(),
                ArgumentType::new(
                    "pollInterval".to_string(),
                    GraphQLType::Scalar(BuiltinScalars::int()),
                )
                .with_default_value(crate::ast::Value::IntValue(5000))
                .with_description("Polling interval in milliseconds".to_string()),
            ),
        );

        // Add graph-level subscription
        subscription_type = subscription_type.with_field(
            "graphChanged".to_string(),
            FieldType::new(
                "graphChanged".to_string(),
                GraphQLType::Object(ObjectType::new("GraphChangeEvent".to_string())),
            )
            .with_description("Subscribe to any changes in the RDF graph".to_string())
            .with_argument(
                "graph".to_string(),
                ArgumentType::new("graph".to_string(), GraphQLType::Scalar(RdfScalars::iri()))
                    .with_description(
                        "The named graph to monitor (default graph if not specified)".to_string(),
                    ),
            ),
        );

        // Add transaction subscription
        subscription_type = subscription_type.with_field(
            "transactionCompleted".to_string(),
            FieldType::new(
                "transactionCompleted".to_string(),
                GraphQLType::Object(ObjectType::new("TransactionEvent".to_string())),
            )
            .with_description("Subscribe to transaction completion events".to_string()),
        );

        Ok(subscription_type)
    }
}

impl Default for SchemaGenerator {
    fn default() -> Self {
        Self::new()
    }
}
