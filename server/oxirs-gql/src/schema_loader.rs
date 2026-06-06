//! Vocabulary extraction and ontology loading for [`SchemaGenerator`].
//!
//! Implements the methods that populate an
//! [`RdfVocabulary`](crate::schema_types::RdfVocabulary): SPARQL-based
//! extraction from an [`RdfStore`](crate::RdfStore), loading and parsing of
//! remote/local ontology documents, and a mock vocabulary used for
//! demonstration purposes.

use crate::schema_generator::SchemaGenerator;
use crate::schema_types::{PropertyType, RdfClass, RdfProperty, RdfVocabulary};
use anyhow::Result;
use oxirs_core::format::{RdfFormat, RdfParser};
use std::collections::HashMap;

impl SchemaGenerator {
    /// Extract RDF vocabulary from a store using SPARQL queries
    pub fn extract_vocabulary_from_store(&self, store: &crate::RdfStore) -> Result<RdfVocabulary> {
        let mut classes = HashMap::new();
        let mut properties = HashMap::new();
        let mut namespaces = HashMap::new();

        // Extract namespaces
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

    fn process_class_results(
        &self,
        results: oxirs_core::query::QueryResults,
        classes: &mut HashMap<String, RdfClass>,
    ) -> Result<()> {
        use oxirs_core::query::QueryResults;

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

                    // Get or create class entry
                    let rdf_class = classes
                        .entry(class_uri.clone())
                        .or_insert_with(|| RdfClass {
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

    fn process_property_results(
        &self,
        results: oxirs_core::query::QueryResults,
        properties: &mut HashMap<String, RdfProperty>,
    ) -> Result<()> {
        use oxirs_core::query::QueryResults;

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
                        .map(|t| t.to_string())
                        .and_then(|s| self.extract_literal_value(&s))
                        .unwrap_or_else(|| "AnnotationProperty".to_string());

                    let prop_type = match property_type.as_str() {
                        "DataProperty" => PropertyType::DataProperty,
                        "ObjectProperty" => PropertyType::ObjectProperty,
                        _ => PropertyType::AnnotationProperty,
                    };

                    // Get or create property entry
                    let rdf_property =
                        properties
                            .entry(property_uri.clone())
                            .or_insert_with(|| RdfProperty {
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
        if let Some(stripped) = term_str.strip_prefix('"') {
            if let Some(end_quote) = stripped.find('"') {
                return Some(stripped[..end_quote].to_string());
            }
        }
        None
    }

    fn link_properties_to_classes(
        &self,
        classes: &mut HashMap<String, RdfClass>,
        properties: &HashMap<String, RdfProperty>,
    ) {
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

    /// Load and parse RDF ontology from URI
    pub(crate) async fn load_ontology_from_uri(&self, ontology_uri: &str) -> Result<RdfVocabulary> {
        // Create a temporary store to load the ontology
        let store = crate::RdfStore::new()?;

        // Determine format based on URI or default to RDF/XML
        let format = self.detect_rdf_format(ontology_uri);

        // Fetch ontology content from URI
        let content = self.fetch_ontology_content(ontology_uri).await?;

        // Parse the RDF content into the store using format-specific parsing
        let parser = RdfParser::new(format);

        // Insert parsed quads into the store
        for quad_result in parser.for_slice(&content) {
            match quad_result {
                Ok(quad) => {
                    store.insert(&quad)?;
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Failed to parse quad from {}: {}",
                        ontology_uri,
                        e
                    ));
                }
            }
        }

        // Extract vocabulary from the loaded ontology using existing method
        self.extract_vocabulary_from_store(&store)
    }

    /// Fetch ontology content from URI (HTTP/HTTPS or local file)
    async fn fetch_ontology_content(&self, uri: &str) -> Result<Vec<u8>> {
        if uri.starts_with("http://") || uri.starts_with("https://") {
            // Fetch from HTTP/HTTPS
            self.fetch_http_content(uri).await
        } else if uri.starts_with("file://") || !uri.contains("://") {
            // Load from local file
            let file_path = if let Some(stripped) = uri.strip_prefix("file://") {
                stripped // Remove "file://" prefix
            } else {
                uri
            };

            match std::fs::read(file_path) {
                Ok(content) => Ok(content),
                Err(e) => Err(anyhow::anyhow!(
                    "Failed to read local file {}: {}",
                    file_path,
                    e
                )),
            }
        } else {
            Err(anyhow::anyhow!("Unsupported URI scheme: {}", uri))
        }
    }

    /// Fetch content from HTTP/HTTPS URI
    async fn fetch_http_content(&self, uri: &str) -> Result<Vec<u8>> {
        use reqwest;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        let response = client
            .get(uri)
            .header(
                "Accept",
                "application/rdf+xml, text/turtle, application/n-triples, application/ld+json",
            )
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "HTTP error {}: Failed to fetch ontology from {}",
                response.status(),
                uri
            ));
        }

        let content = response.bytes().await?;
        Ok(content.to_vec())
    }

    /// Detect RDF format based on URI or content-type
    fn detect_rdf_format(&self, uri: &str) -> RdfFormat {
        use oxirs_core::format::JsonLdProfileSet;

        let uri_lower = uri.to_lowercase();

        if uri_lower.ends_with(".ttl") || uri_lower.ends_with(".turtle") {
            RdfFormat::Turtle
        } else if uri_lower.ends_with(".nt") || uri_lower.ends_with(".ntriples") {
            RdfFormat::NTriples
        } else if uri_lower.ends_with(".jsonld") || uri_lower.ends_with(".json-ld") {
            RdfFormat::JsonLd {
                profile: JsonLdProfileSet::empty(),
            }
        } else if uri_lower.ends_with(".n3") {
            RdfFormat::N3
        } else {
            // Default to RDF/XML for .rdf, .owl, or unknown extensions
            RdfFormat::RdfXml
        }
    }

    #[allow(dead_code)]
    fn load_mock_vocabulary(&self, _ontology_uri: &str) -> Result<RdfVocabulary> {
        // Enhanced mock vocabulary for demonstration
        let mut classes = HashMap::new();
        let mut properties = HashMap::new();
        let mut namespaces = HashMap::new();

        // Common namespaces
        namespaces.insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());
        namespaces.insert("schema".to_string(), "http://schema.org/".to_string());
        namespaces.insert(
            "dbo".to_string(),
            "http://dbpedia.org/ontology/".to_string(),
        );
        namespaces.insert(
            "dc".to_string(),
            "http://purl.org/dc/elements/1.1/".to_string(),
        );

        // FOAF Agent (base class)
        classes.insert(
            "http://xmlns.com/foaf/0.1/Agent".to_string(),
            RdfClass {
                uri: "http://xmlns.com/foaf/0.1/Agent".to_string(),
                label: Some("Agent".to_string()),
                comment: Some(
                    "An agent (eg. person, group, software or physical artifact)".to_string(),
                ),
                super_classes: vec![],
                properties: vec!["http://xmlns.com/foaf/0.1/name".to_string()],
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
            (
                "http://xmlns.com/foaf/0.1/name",
                "name",
                "A name for some thing",
                PropertyType::DataProperty,
                vec!["http://xmlns.com/foaf/0.1/Agent"],
                vec!["http://www.w3.org/2001/XMLSchema#string"],
            ),
            (
                "http://xmlns.com/foaf/0.1/email",
                "email",
                "An email address",
                PropertyType::DataProperty,
                vec!["http://xmlns.com/foaf/0.1/Person"],
                vec!["http://www.w3.org/2001/XMLSchema#string"],
            ),
            (
                "http://xmlns.com/foaf/0.1/age",
                "age",
                "The age in years of some agent",
                PropertyType::DataProperty,
                vec!["http://xmlns.com/foaf/0.1/Person"],
                vec!["http://www.w3.org/2001/XMLSchema#int"],
            ),
            (
                "http://xmlns.com/foaf/0.1/homepage",
                "homepage",
                "A homepage for some thing",
                PropertyType::DataProperty,
                vec!["http://xmlns.com/foaf/0.1/Agent"],
                vec!["http://www.w3.org/2001/XMLSchema#anyURI"],
            ),
            (
                "http://xmlns.com/foaf/0.1/knows",
                "knows",
                "A person known by this person",
                PropertyType::ObjectProperty,
                vec!["http://xmlns.com/foaf/0.1/Person"],
                vec!["http://xmlns.com/foaf/0.1/Person"],
            ),
            (
                "http://schema.org/name",
                "name",
                "The name of the item",
                PropertyType::DataProperty,
                vec!["http://schema.org/Product"],
                vec!["http://www.w3.org/2001/XMLSchema#string"],
            ),
            (
                "http://schema.org/description",
                "description",
                "A description of the item",
                PropertyType::DataProperty,
                vec!["http://schema.org/Product"],
                vec!["http://www.w3.org/2001/XMLSchema#string"],
            ),
            (
                "http://schema.org/price",
                "price",
                "The price of the product",
                PropertyType::DataProperty,
                vec!["http://schema.org/Product"],
                vec!["http://www.w3.org/2001/XMLSchema#decimal"],
            ),
            (
                "http://schema.org/manufacturer",
                "manufacturer",
                "The manufacturer of the product",
                PropertyType::ObjectProperty,
                vec!["http://schema.org/Product"],
                vec!["http://xmlns.com/foaf/0.1/Organization"],
            ),
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
