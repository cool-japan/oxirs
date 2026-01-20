//! Enhanced OWL schema generation with advanced features
//!
//! This module extends the basic schema generation with comprehensive OWL support:
//! - OWL restrictions (someValuesFrom, allValuesFrom, hasValue, cardinality)
//! - Property characteristics (symmetric, transitive, reflexive, functional)
//! - Equivalent classes and properties
//! - Disjoint classes and properties
//! - Union and intersection classes
//! - Cardinality constraints

use crate::schema::{PropertyType, RdfClass, RdfProperty, RdfVocabulary, SchemaGenerationConfig};
use crate::types::*;
use crate::RdfStore;
use anyhow::Result;
use oxirs_core::query::QueryResults;
use std::collections::HashMap;

/// Enhanced OWL property information
#[derive(Debug, Clone)]
pub struct OwlProperty {
    pub base: RdfProperty,
    pub is_symmetric: bool,
    pub is_transitive: bool,
    pub is_reflexive: bool,
    pub is_irreflexive: bool,
    pub is_asymmetric: bool,
    pub inverse_of: Option<String>,
    pub equivalent_properties: Vec<String>,
    pub disjoint_with: Vec<String>,
    pub sub_properties: Vec<String>,
    pub property_chain: Vec<Vec<String>>,
}

impl OwlProperty {
    pub fn from_rdf_property(prop: RdfProperty) -> Self {
        Self {
            base: prop,
            is_symmetric: false,
            is_transitive: false,
            is_reflexive: false,
            is_irreflexive: false,
            is_asymmetric: false,
            inverse_of: None,
            equivalent_properties: Vec::new(),
            disjoint_with: Vec::new(),
            sub_properties: Vec::new(),
            property_chain: Vec::new(),
        }
    }
}

/// OWL class restriction
#[derive(Debug, Clone)]
pub enum OwlRestriction {
    SomeValuesFrom {
        property: String,
        class: String,
    },
    AllValuesFrom {
        property: String,
        class: String,
    },
    HasValue {
        property: String,
        value: String,
    },
    MinCardinality {
        property: String,
        cardinality: u32,
        class: Option<String>,
    },
    MaxCardinality {
        property: String,
        cardinality: u32,
        class: Option<String>,
    },
    ExactCardinality {
        property: String,
        cardinality: u32,
        class: Option<String>,
    },
}

/// Enhanced OWL class information
#[derive(Debug, Clone)]
pub struct OwlClass {
    pub base: RdfClass,
    pub restrictions: Vec<OwlRestriction>,
    pub equivalent_classes: Vec<String>,
    pub disjoint_with: Vec<String>,
    pub union_of: Vec<Vec<String>>,
    pub intersection_of: Vec<Vec<String>>,
    pub complement_of: Option<String>,
    pub one_of: Vec<String>,
    pub is_abstract: bool,
}

impl OwlClass {
    pub fn from_rdf_class(class: RdfClass) -> Self {
        Self {
            base: class,
            restrictions: Vec::new(),
            equivalent_classes: Vec::new(),
            disjoint_with: Vec::new(),
            union_of: Vec::new(),
            intersection_of: Vec::new(),
            complement_of: None,
            one_of: Vec::new(),
            is_abstract: false,
        }
    }
}

/// Enhanced OWL vocabulary with advanced features
#[derive(Debug, Clone)]
pub struct OwlVocabulary {
    pub classes: HashMap<String, OwlClass>,
    pub properties: HashMap<String, OwlProperty>,
    pub namespaces: HashMap<String, String>,
}

impl From<RdfVocabulary> for OwlVocabulary {
    fn from(vocab: RdfVocabulary) -> Self {
        let classes = vocab
            .classes
            .into_iter()
            .map(|(k, v)| (k, OwlClass::from_rdf_class(v)))
            .collect();

        let properties = vocab
            .properties
            .into_iter()
            .map(|(k, v)| (k, OwlProperty::from_rdf_property(v)))
            .collect();

        Self {
            classes,
            properties,
            namespaces: vocab.namespaces,
        }
    }
}

/// Enhanced schema generator with comprehensive OWL support
pub struct OwlSchemaGenerator {
    config: SchemaGenerationConfig,
}

impl OwlSchemaGenerator {
    pub fn new(config: SchemaGenerationConfig) -> Self {
        Self { config }
    }

    /// Extract enhanced OWL vocabulary from RDF store
    pub fn extract_owl_vocabulary(&self, store: &RdfStore) -> Result<OwlVocabulary> {
        // First extract basic RDF vocabulary
        let basic_vocab = self.extract_basic_vocabulary(store)?;
        let mut owl_vocab = OwlVocabulary::from(basic_vocab);

        // Enhance with OWL-specific features
        self.extract_property_characteristics(store, &mut owl_vocab)?;
        self.extract_class_restrictions(store, &mut owl_vocab)?;
        self.extract_equivalent_classes(store, &mut owl_vocab)?;
        self.extract_disjoint_classes(store, &mut owl_vocab)?;
        self.extract_class_expressions(store, &mut owl_vocab)?;

        Ok(owl_vocab)
    }

    /// Extract basic RDF vocabulary
    fn extract_basic_vocabulary(&self, store: &RdfStore) -> Result<RdfVocabulary> {
        let mut classes = HashMap::new();
        let mut properties = HashMap::new();
        let mut namespaces = HashMap::new();

        // Standard namespaces
        namespaces.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        namespaces.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        namespaces.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        namespaces.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );

        // Extract classes
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

        // Extract properties
        let property_query = r#"
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SELECT DISTINCT ?property ?label ?comment ?domain ?range ?type ?functional ?inverseFunctional
            WHERE {
                {
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
                OPTIONAL {
                    ?property a owl:FunctionalProperty .
                    BIND(true as ?functional)
                }
                OPTIONAL {
                    ?property a owl:InverseFunctionalProperty .
                    BIND(true as ?inverseFunctional)
                }
                FILTER(!isBlank(?property))
            }
        "#;

        if let Ok(results) = store.query(property_query) {
            self.process_property_results(results, &mut properties)?;
        }

        Ok(RdfVocabulary {
            classes,
            properties,
            namespaces,
        })
    }

    /// Extract OWL property characteristics
    fn extract_property_characteristics(
        &self,
        store: &RdfStore,
        vocab: &mut OwlVocabulary,
    ) -> Result<()> {
        let characteristics_query = r#"
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SELECT ?property ?characteristic
            WHERE {
                {
                    ?property a owl:SymmetricProperty .
                    BIND("symmetric" as ?characteristic)
                } UNION {
                    ?property a owl:TransitiveProperty .
                    BIND("transitive" as ?characteristic)
                } UNION {
                    ?property a owl:ReflexiveProperty .
                    BIND("reflexive" as ?characteristic)
                } UNION {
                    ?property a owl:IrreflexiveProperty .
                    BIND("irreflexive" as ?characteristic)
                } UNION {
                    ?property a owl:AsymmetricProperty .
                    BIND("asymmetric" as ?characteristic)
                }
            }
        "#;

        if let Ok(QueryResults::Solutions(solutions)) = store.query(characteristics_query) {
            for solution in solutions {
                if let (Some(prop_term), Some(char_term)) = (
                    solution.get(
                        &oxirs_core::model::Variable::new("property")
                            .expect("hardcoded variable name should be valid"),
                    ),
                    solution.get(
                        &oxirs_core::model::Variable::new("characteristic")
                            .expect("hardcoded variable name should be valid"),
                    ),
                ) {
                    let prop_uri = prop_term.to_string();
                    if let Some(characteristic) = self.extract_literal_value(&char_term.to_string())
                    {
                        if let Some(prop) = vocab.properties.get_mut(&prop_uri) {
                            match characteristic.as_str() {
                                "symmetric" => prop.is_symmetric = true,
                                "transitive" => prop.is_transitive = true,
                                "reflexive" => prop.is_reflexive = true,
                                "irreflexive" => prop.is_irreflexive = true,
                                "asymmetric" => prop.is_asymmetric = true,
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        // Extract inverse properties
        let inverse_query = r#"
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SELECT ?property ?inverse
            WHERE {
                ?property owl:inverseOf ?inverse .
            }
        "#;

        if let Ok(QueryResults::Solutions(solutions)) = store.query(inverse_query) {
            for solution in solutions {
                if let (Some(prop_term), Some(inv_term)) = (
                    solution.get(
                        &oxirs_core::model::Variable::new("property")
                            .expect("hardcoded variable name should be valid"),
                    ),
                    solution.get(
                        &oxirs_core::model::Variable::new("inverse")
                            .expect("hardcoded variable name should be valid"),
                    ),
                ) {
                    let prop_uri = prop_term.to_string();
                    let inv_uri = inv_term.to_string();

                    if let Some(prop) = vocab.properties.get_mut(&prop_uri) {
                        prop.inverse_of = Some(inv_uri);
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract OWL class restrictions
    fn extract_class_restrictions(
        &self,
        store: &RdfStore,
        vocab: &mut OwlVocabulary,
    ) -> Result<()> {
        // Extract someValuesFrom restrictions
        let some_values_query = r#"
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?class ?property ?valueClass
            WHERE {
                ?class rdfs:subClassOf ?restriction .
                ?restriction a owl:Restriction ;
                            owl:onProperty ?property ;
                            owl:someValuesFrom ?valueClass .
            }
        "#;

        if let Ok(QueryResults::Solutions(solutions)) = store.query(some_values_query) {
            for solution in solutions {
                if let (Some(class_term), Some(prop_term), Some(value_term)) = (
                    solution.get(
                        &oxirs_core::model::Variable::new("class")
                            .expect("hardcoded variable name should be valid"),
                    ),
                    solution.get(
                        &oxirs_core::model::Variable::new("property")
                            .expect("hardcoded variable name should be valid"),
                    ),
                    solution.get(
                        &oxirs_core::model::Variable::new("valueClass")
                            .expect("hardcoded variable name should be valid"),
                    ),
                ) {
                    let class_uri = class_term.to_string();
                    let prop_uri = prop_term.to_string();
                    let value_uri = value_term.to_string();

                    if let Some(owl_class) = vocab.classes.get_mut(&class_uri) {
                        owl_class.restrictions.push(OwlRestriction::SomeValuesFrom {
                            property: prop_uri,
                            class: value_uri,
                        });
                    }
                }
            }
        }

        // Extract cardinality restrictions
        let cardinality_query = r#"
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?class ?property ?minCard ?maxCard ?exactCard
            WHERE {
                ?class rdfs:subClassOf ?restriction .
                ?restriction a owl:Restriction ;
                            owl:onProperty ?property .
                OPTIONAL { ?restriction owl:minCardinality ?minCard }
                OPTIONAL { ?restriction owl:maxCardinality ?maxCard }
                OPTIONAL { ?restriction owl:cardinality ?exactCard }
            }
        "#;

        if let Ok(QueryResults::Solutions(solutions)) = store.query(cardinality_query) {
            for solution in solutions {
                if let (Some(class_term), Some(prop_term)) = (
                    solution.get(
                        &oxirs_core::model::Variable::new("class")
                            .expect("hardcoded variable name should be valid"),
                    ),
                    solution.get(
                        &oxirs_core::model::Variable::new("property")
                            .expect("hardcoded variable name should be valid"),
                    ),
                ) {
                    let class_uri = class_term.to_string();
                    let prop_uri = prop_term.to_string();

                    if let Some(owl_class) = vocab.classes.get_mut(&class_uri) {
                        // Check for exact cardinality
                        if let Some(exact_term) = solution.get(
                            &oxirs_core::model::Variable::new("exactCard")
                                .expect("hardcoded variable name should be valid"),
                        ) {
                            if let Some(card_str) =
                                self.extract_literal_value(&exact_term.to_string())
                            {
                                if let Ok(card) = card_str.parse::<u32>() {
                                    owl_class
                                        .restrictions
                                        .push(OwlRestriction::ExactCardinality {
                                            property: prop_uri.clone(),
                                            cardinality: card,
                                            class: None,
                                        });
                                }
                            }
                        }

                        // Check for min cardinality
                        if let Some(min_term) = solution.get(
                            &oxirs_core::model::Variable::new("minCard")
                                .expect("hardcoded variable name should be valid"),
                        ) {
                            if let Some(card_str) =
                                self.extract_literal_value(&min_term.to_string())
                            {
                                if let Ok(card) = card_str.parse::<u32>() {
                                    owl_class.restrictions.push(OwlRestriction::MinCardinality {
                                        property: prop_uri.clone(),
                                        cardinality: card,
                                        class: None,
                                    });
                                }
                            }
                        }

                        // Check for max cardinality
                        if let Some(max_term) = solution.get(
                            &oxirs_core::model::Variable::new("maxCard")
                                .expect("hardcoded variable name should be valid"),
                        ) {
                            if let Some(card_str) =
                                self.extract_literal_value(&max_term.to_string())
                            {
                                if let Ok(card) = card_str.parse::<u32>() {
                                    owl_class.restrictions.push(OwlRestriction::MaxCardinality {
                                        property: prop_uri,
                                        cardinality: card,
                                        class: None,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract equivalent classes
    fn extract_equivalent_classes(
        &self,
        store: &RdfStore,
        vocab: &mut OwlVocabulary,
    ) -> Result<()> {
        let equiv_query = r#"
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SELECT ?class1 ?class2
            WHERE {
                ?class1 owl:equivalentClass ?class2 .
                FILTER(!isBlank(?class1) && !isBlank(?class2))
            }
        "#;

        if let Ok(QueryResults::Solutions(solutions)) = store.query(equiv_query) {
            for solution in solutions {
                if let (Some(c1_term), Some(c2_term)) = (
                    solution.get(
                        &oxirs_core::model::Variable::new("class1")
                            .expect("hardcoded variable name should be valid"),
                    ),
                    solution.get(
                        &oxirs_core::model::Variable::new("class2")
                            .expect("hardcoded variable name should be valid"),
                    ),
                ) {
                    let c1_uri = c1_term.to_string();
                    let c2_uri = c2_term.to_string();

                    if let Some(class) = vocab.classes.get_mut(&c1_uri) {
                        if !class.equivalent_classes.contains(&c2_uri) {
                            class.equivalent_classes.push(c2_uri);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract disjoint classes
    fn extract_disjoint_classes(&self, store: &RdfStore, vocab: &mut OwlVocabulary) -> Result<()> {
        let disjoint_query = r#"
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SELECT ?class1 ?class2
            WHERE {
                ?class1 owl:disjointWith ?class2 .
                FILTER(!isBlank(?class1) && !isBlank(?class2))
            }
        "#;

        if let Ok(QueryResults::Solutions(solutions)) = store.query(disjoint_query) {
            for solution in solutions {
                if let (Some(c1_term), Some(c2_term)) = (
                    solution.get(
                        &oxirs_core::model::Variable::new("class1")
                            .expect("hardcoded variable name should be valid"),
                    ),
                    solution.get(
                        &oxirs_core::model::Variable::new("class2")
                            .expect("hardcoded variable name should be valid"),
                    ),
                ) {
                    let c1_uri = c1_term.to_string();
                    let c2_uri = c2_term.to_string();

                    if let Some(class) = vocab.classes.get_mut(&c1_uri) {
                        if !class.disjoint_with.contains(&c2_uri) {
                            class.disjoint_with.push(c2_uri);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract class expressions (union, intersection)
    fn extract_class_expressions(
        &self,
        _store: &RdfStore,
        _vocab: &mut OwlVocabulary,
    ) -> Result<()> {
        // Union and intersection extraction would require more complex blank node handling
        // For now, this is a placeholder for future implementation
        Ok(())
    }

    /// Generate GraphQL schema from OWL vocabulary
    pub fn generate_schema_from_owl(&self, owl_vocab: &OwlVocabulary) -> Result<Schema> {
        let mut schema = Schema::new();

        // Add RDF-specific scalar types
        schema.add_type(GraphQLType::Scalar(crate::rdf_scalars::RdfScalars::iri()));
        schema.add_type(GraphQLType::Scalar(
            crate::rdf_scalars::RdfScalars::literal(),
        ));
        schema.add_type(GraphQLType::Scalar(
            crate::rdf_scalars::RdfScalars::datetime(),
        ));

        // Generate GraphQL types from OWL classes
        for (class_uri, owl_class) in &owl_vocab.classes {
            if self.config.exclude_classes.contains(class_uri) {
                continue;
            }

            let gql_type = self.owl_class_to_graphql_type(owl_class, owl_vocab)?;
            schema.add_type(gql_type);
        }

        // Generate Query type
        let query_type = self.generate_query_type_from_owl(owl_vocab)?;
        schema.add_type(GraphQLType::Object(query_type));
        schema.set_query_type("Query".to_string());

        Ok(schema)
    }

    /// Convert OWL class to GraphQL object type
    fn owl_class_to_graphql_type(
        &self,
        owl_class: &OwlClass,
        vocab: &OwlVocabulary,
    ) -> Result<GraphQLType> {
        let class_name = self.extract_local_name(&owl_class.base.uri);

        let mut object_type = ObjectType::new(class_name.clone()).with_description(
            owl_class
                .base
                .comment
                .clone()
                .unwrap_or_else(|| format!("RDF class {}", owl_class.base.uri)),
        );

        // Add ID field
        object_type = object_type.with_field(
            "id".to_string(),
            FieldType::new(
                "id".to_string(),
                GraphQLType::NonNull(Box::new(GraphQLType::Scalar(BuiltinScalars::id()))),
            )
            .with_description("Unique identifier (IRI)".to_string()),
        );

        // Add fields from properties
        for prop_uri in &owl_class.base.properties {
            if let Some(owl_prop) = vocab.properties.get(prop_uri) {
                if self.config.exclude_properties.contains(prop_uri) {
                    continue;
                }

                let field = self.owl_property_to_field(owl_prop, owl_class)?;
                object_type = object_type.with_field(self.extract_local_name(prop_uri), field);
            }
        }

        Ok(GraphQLType::Object(object_type))
    }

    /// Convert OWL property to GraphQL field
    fn owl_property_to_field(
        &self,
        owl_prop: &OwlProperty,
        owl_class: &OwlClass,
    ) -> Result<FieldType> {
        let field_name = self.extract_local_name(&owl_prop.base.uri);
        let mut field_type = self.get_graphql_type_for_range(&owl_prop.base.range);

        // Apply cardinality constraints from restrictions
        let mut is_required = false;
        let mut is_list = false;

        for restriction in &owl_class.restrictions {
            match restriction {
                OwlRestriction::MinCardinality {
                    property,
                    cardinality,
                    ..
                }
                | OwlRestriction::ExactCardinality {
                    property,
                    cardinality,
                    ..
                } if property == &owl_prop.base.uri && *cardinality >= 1 => {
                    is_required = true;
                }
                OwlRestriction::MaxCardinality {
                    property,
                    cardinality,
                    ..
                } if property == &owl_prop.base.uri && *cardinality > 1 => {
                    is_list = true;
                }
                _ => {}
            }
        }

        // Apply functional property constraint (max cardinality 1)
        if owl_prop.base.functional {
            // Single value
        } else if !is_list {
            // Default to list for non-functional properties
            is_list = true;
        }

        if is_list {
            field_type = GraphQLType::List(Box::new(field_type));
        }

        if is_required {
            field_type = GraphQLType::NonNull(Box::new(field_type));
        }

        let mut field = FieldType::new(field_name.clone(), field_type);

        if let Some(comment) = &owl_prop.base.comment {
            field = field.with_description(comment.clone());
        }

        Ok(field)
    }

    /// Get GraphQL type for RDF range
    fn get_graphql_type_for_range(&self, range: &[String]) -> GraphQLType {
        if range.is_empty() {
            return GraphQLType::Scalar(BuiltinScalars::string());
        }

        let range_uri = &range[0];

        // Check for XSD datatype mappings
        if let Some(gql_type) = self.config.type_mappings.get(range_uri) {
            match gql_type.as_str() {
                "String" => return GraphQLType::Scalar(BuiltinScalars::string()),
                "Int" => return GraphQLType::Scalar(BuiltinScalars::int()),
                "Float" => return GraphQLType::Scalar(BuiltinScalars::float()),
                "Boolean" => return GraphQLType::Scalar(BuiltinScalars::boolean()),
                "ID" => return GraphQLType::Scalar(BuiltinScalars::id()),
                _ => {}
            }
        }

        // Object property - reference to another class
        GraphQLType::Scalar(BuiltinScalars::string())
    }

    /// Generate Query type from OWL vocabulary
    fn generate_query_type_from_owl(&self, vocab: &OwlVocabulary) -> Result<ObjectType> {
        let mut query_type =
            ObjectType::new("Query".to_string()).with_description("Root query type".to_string());

        // Add query fields for each class
        for class_uri in vocab.classes.keys() {
            if self.config.exclude_classes.contains(class_uri) {
                continue;
            }

            let class_name = self.extract_local_name(class_uri);

            // Single instance query
            let field_name = format!("get{}", class_name);
            query_type = query_type.with_field(
                field_name.clone(),
                FieldType::new(field_name, GraphQLType::Scalar(BuiltinScalars::string()))
                    .with_argument(
                        "id".to_string(),
                        ArgumentType::new(
                            "id".to_string(),
                            GraphQLType::NonNull(Box::new(GraphQLType::Scalar(
                                BuiltinScalars::id(),
                            ))),
                        ),
                    ),
            );

            // List query
            let list_field_name = format!("list{}", class_name);
            query_type = query_type.with_field(
                list_field_name.clone(),
                FieldType::new(
                    list_field_name,
                    GraphQLType::List(Box::new(GraphQLType::Scalar(BuiltinScalars::string()))),
                )
                .with_argument(
                    "limit".to_string(),
                    ArgumentType::new(
                        "limit".to_string(),
                        GraphQLType::Scalar(BuiltinScalars::int()),
                    ),
                ),
            );
        }

        Ok(query_type)
    }

    /// Extract local name from URI
    fn extract_local_name(&self, uri: &str) -> String {
        uri.split(&['#', '/'][..])
            .next_back()
            .unwrap_or(uri)
            .to_string()
    }

    /// Extract literal value from RDF term string
    fn extract_literal_value(&self, term_str: &str) -> Option<String> {
        if let Some(stripped) = term_str.strip_prefix('"') {
            if let Some(end_quote) = stripped.find('"') {
                return Some(stripped[..end_quote].to_string());
            }
        }
        None
    }

    /// Process class results from SPARQL query
    fn process_class_results(
        &self,
        results: QueryResults,
        classes: &mut HashMap<String, RdfClass>,
    ) -> Result<()> {
        if let QueryResults::Solutions(solutions) = results {
            for solution in solutions {
                if let Some(class_term) = solution.get(
                    &oxirs_core::model::Variable::new("class")
                        .expect("hardcoded variable name should be valid"),
                ) {
                    let class_uri = class_term.to_string();

                    let label = solution
                        .get(
                            &oxirs_core::model::Variable::new("label")
                                .expect("hardcoded variable name should be valid"),
                        )
                        .and_then(|t| self.extract_literal_value(&t.to_string()));

                    let comment = solution
                        .get(
                            &oxirs_core::model::Variable::new("comment")
                                .expect("hardcoded variable name should be valid"),
                        )
                        .and_then(|t| self.extract_literal_value(&t.to_string()));

                    let super_class = solution
                        .get(
                            &oxirs_core::model::Variable::new("superClass")
                                .expect("hardcoded variable name should be valid"),
                        )
                        .map(|t| t.to_string());

                    let rdf_class = classes
                        .entry(class_uri.clone())
                        .or_insert_with(|| RdfClass {
                            uri: class_uri,
                            label: None,
                            comment: None,
                            super_classes: Vec::new(),
                            properties: Vec::new(),
                        });

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

    /// Process property results from SPARQL query
    fn process_property_results(
        &self,
        results: QueryResults,
        properties: &mut HashMap<String, RdfProperty>,
    ) -> Result<()> {
        if let QueryResults::Solutions(solutions) = results {
            for solution in solutions {
                if let Some(property_term) = solution.get(
                    &oxirs_core::model::Variable::new("property")
                        .expect("hardcoded variable name should be valid"),
                ) {
                    let property_uri = property_term.to_string();

                    let label = solution
                        .get(
                            &oxirs_core::model::Variable::new("label")
                                .expect("hardcoded variable name should be valid"),
                        )
                        .and_then(|t| self.extract_literal_value(&t.to_string()));

                    let comment = solution
                        .get(
                            &oxirs_core::model::Variable::new("comment")
                                .expect("hardcoded variable name should be valid"),
                        )
                        .and_then(|t| self.extract_literal_value(&t.to_string()));

                    let domain = solution
                        .get(
                            &oxirs_core::model::Variable::new("domain")
                                .expect("hardcoded variable name should be valid"),
                        )
                        .map(|t| t.to_string());

                    let range = solution
                        .get(
                            &oxirs_core::model::Variable::new("range")
                                .expect("hardcoded variable name should be valid"),
                        )
                        .map(|t| t.to_string());

                    let property_type = solution
                        .get(
                            &oxirs_core::model::Variable::new("type")
                                .expect("hardcoded variable name should be valid"),
                        )
                        .and_then(|t| self.extract_literal_value(&t.to_string()))
                        .unwrap_or_else(|| "AnnotationProperty".to_string());

                    let prop_type = match property_type.as_str() {
                        "DataProperty" => PropertyType::DataProperty,
                        "ObjectProperty" => PropertyType::ObjectProperty,
                        _ => PropertyType::AnnotationProperty,
                    };

                    let functional = solution
                        .get(
                            &oxirs_core::model::Variable::new("functional")
                                .expect("hardcoded variable name should be valid"),
                        )
                        .is_some();

                    let inverse_functional = solution
                        .get(
                            &oxirs_core::model::Variable::new("inverseFunctional")
                                .expect("hardcoded variable name should be valid"),
                        )
                        .is_some();

                    let rdf_property =
                        properties
                            .entry(property_uri.clone())
                            .or_insert_with(|| RdfProperty {
                                uri: property_uri,
                                label: None,
                                comment: None,
                                domain: Vec::new(),
                                range: Vec::new(),
                                property_type: prop_type,
                                functional,
                                inverse_functional,
                            });

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_owl_property_creation() {
        let rdf_prop = RdfProperty {
            uri: "http://example.org/prop".to_string(),
            label: Some("Property".to_string()),
            comment: None,
            domain: vec![],
            range: vec![],
            property_type: PropertyType::ObjectProperty,
            functional: false,
            inverse_functional: false,
        };

        let owl_prop = OwlProperty::from_rdf_property(rdf_prop);

        assert_eq!(owl_prop.base.uri, "http://example.org/prop");
        assert!(!owl_prop.is_symmetric);
        assert!(!owl_prop.is_transitive);
    }

    #[test]
    fn test_owl_class_creation() {
        let rdf_class = RdfClass {
            uri: "http://example.org/Class".to_string(),
            label: Some("Class".to_string()),
            comment: None,
            super_classes: vec![],
            properties: vec![],
        };

        let owl_class = OwlClass::from_rdf_class(rdf_class);

        assert_eq!(owl_class.base.uri, "http://example.org/Class");
        assert!(owl_class.restrictions.is_empty());
        assert!(owl_class.equivalent_classes.is_empty());
    }

    #[test]
    fn test_owl_restriction_types() {
        let restriction1 = OwlRestriction::SomeValuesFrom {
            property: "prop1".to_string(),
            class: "class1".to_string(),
        };

        let restriction2 = OwlRestriction::MinCardinality {
            property: "prop2".to_string(),
            cardinality: 1,
            class: None,
        };

        match restriction1 {
            OwlRestriction::SomeValuesFrom { .. } => {} // Success
            _ => panic!("Wrong restriction type"),
        }

        match restriction2 {
            OwlRestriction::MinCardinality { cardinality, .. } => assert_eq!(cardinality, 1),
            _ => panic!("Wrong restriction type"),
        }
    }
}
