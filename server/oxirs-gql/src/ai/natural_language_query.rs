// Copyright (c) 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Natural Language Query Generation
//!
//! This module provides AI-powered natural language to GraphQL query translation,
//! enabling users to write queries in plain English that are automatically
//! converted to valid GraphQL syntax.

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Natural language query input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalLanguageQuery {
    /// The natural language query text
    pub text: String,
    /// Optional context for disambiguation
    pub context: Option<String>,
    /// Confidence threshold for generation
    pub confidence_threshold: f32,
}

impl NaturalLanguageQuery {
    /// Create a new natural language query
    pub fn new(text: String) -> Self {
        Self {
            text,
            context: None,
            confidence_threshold: 0.7,
        }
    }

    /// Add context to the query
    pub fn with_context(mut self, context: String) -> Self {
        self.context = Some(context);
        self
    }

    /// Set confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }
}

/// Generated GraphQL query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedQuery {
    /// The generated GraphQL query
    pub query: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Alternative interpretations
    pub alternatives: Vec<AlternativeQuery>,
    /// Extracted entities and intent
    pub metadata: QueryMetadata,
}

/// Alternative query interpretation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeQuery {
    /// Alternative GraphQL query
    pub query: String,
    /// Confidence score for this alternative
    pub confidence: f32,
    /// Explanation of interpretation
    pub explanation: String,
}

/// Query metadata extracted from natural language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetadata {
    /// Detected intent (e.g., "search", "filter", "aggregate")
    pub intent: String,
    /// Extracted entities (field names, values, etc.)
    pub entities: HashMap<String, String>,
    /// Detected operations
    pub operations: Vec<String>,
    /// Suggested fields
    pub suggested_fields: Vec<String>,
}

impl Default for QueryMetadata {
    fn default() -> Self {
        Self {
            intent: "unknown".to_string(),
            entities: HashMap::new(),
            operations: Vec::new(),
            suggested_fields: Vec::new(),
        }
    }
}

/// Schema information for query generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaInfo {
    /// Available types
    pub types: Vec<String>,
    /// Available fields per type
    pub fields: HashMap<String, Vec<String>>,
    /// Field descriptions
    pub descriptions: HashMap<String, String>,
}

impl SchemaInfo {
    /// Create a new schema info
    pub fn new() -> Self {
        Self {
            types: Vec::new(),
            fields: HashMap::new(),
            descriptions: HashMap::new(),
        }
    }

    /// Add a type to the schema
    pub fn add_type(&mut self, type_name: String, fields: Vec<String>) {
        self.types.push(type_name.clone());
        self.fields.insert(type_name, fields);
    }

    /// Add field description
    pub fn add_description(&mut self, field: String, description: String) {
        self.descriptions.insert(field, description);
    }
}

impl Default for SchemaInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Natural language query generator using ML
pub struct NaturalLanguageQueryGenerator {
    /// Schema information
    schema: Arc<RwLock<SchemaInfo>>,
    /// Intent classifier (simulated with embeddings)
    intent_classifier: Arc<RwLock<IntentClassifier>>,
    /// Entity extractor
    entity_extractor: Arc<RwLock<EntityExtractor>>,
    /// Query template repository
    templates: Arc<RwLock<Vec<QueryTemplate>>>,
}

/// Intent classifier for understanding query purpose
#[derive(Debug, Clone)]
pub struct IntentClassifier {
    /// Intent embeddings (intent name -> embedding vector)
    intents: HashMap<String, Array1<f32>>,
}

impl IntentClassifier {
    /// Create a new intent classifier
    pub fn new() -> Self {
        let mut classifier = Self {
            intents: HashMap::new(),
        };
        classifier.initialize_intents();
        classifier
    }

    /// Initialize common intents
    fn initialize_intents(&mut self) {
        let embedding_dim = 128;
        let intents = vec![
            "search",
            "filter",
            "aggregate",
            "count",
            "list",
            "get",
            "find",
            "sort",
            "group",
            "update",
            "delete",
            "create",
        ];

        for intent in intents {
            // Generate deterministic embedding for each intent (in production, use pre-trained)
            let embedding = Array1::from_vec(
                (0..embedding_dim)
                    .map(|i| ((i as f32 * 0.1) % 2.0) - 1.0)
                    .collect(),
            );
            self.intents.insert(intent.to_string(), embedding);
        }
    }

    /// Classify query intent
    pub fn classify(&self, text: &str) -> (String, f32) {
        // Simple keyword-based classification (in production, use NLP model)
        let text_lower = text.to_lowercase();

        let mut best_intent = "search".to_string();
        let mut best_score = 0.5;

        for intent in self.intents.keys() {
            let score = if text_lower.contains(intent) {
                0.9
            } else {
                0.3
            };

            if score > best_score {
                best_score = score;
                best_intent = intent.clone();
            }
        }

        (best_intent, best_score)
    }
}

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Entity extractor for identifying fields and values
#[derive(Debug, Clone)]
pub struct EntityExtractor {
    /// Field name patterns
    #[allow(dead_code)]
    patterns: HashMap<String, Vec<String>>,
}

impl EntityExtractor {
    /// Create a new entity extractor
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }

    /// Extract entities from text
    pub fn extract(&self, text: &str, schema: &SchemaInfo) -> HashMap<String, String> {
        let mut entities = HashMap::new();
        let text_lower = text.to_lowercase();

        // Extract field names mentioned in the query
        for type_name in &schema.types {
            if let Some(fields) = schema.fields.get(type_name) {
                for field in fields {
                    if text_lower.contains(&field.to_lowercase()) {
                        entities.insert(field.clone(), type_name.clone());
                    }
                }
            }
        }

        // Extract common patterns (e.g., "name is John" -> {name: "John"})
        let words: Vec<&str> = text.split_whitespace().collect();
        for i in 0..words.len().saturating_sub(2) {
            if words[i + 1] == "is" || words[i + 1] == "equals" {
                entities.insert(
                    words[i].to_string(),
                    words[i + 2].trim_matches('"').to_string(),
                );
            }
        }

        entities
    }
}

impl Default for EntityExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Query template for generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTemplate {
    /// Template name
    pub name: String,
    /// Intent this template is for
    pub intent: String,
    /// GraphQL template string
    pub template: String,
    /// Required entities
    pub required_entities: Vec<String>,
}

impl NaturalLanguageQueryGenerator {
    /// Create a new natural language query generator
    pub fn new() -> Self {
        Self {
            schema: Arc::new(RwLock::new(SchemaInfo::new())),
            intent_classifier: Arc::new(RwLock::new(IntentClassifier::new())),
            entity_extractor: Arc::new(RwLock::new(EntityExtractor::new())),
            templates: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register schema information
    pub async fn register_schema(&self, schema: SchemaInfo) -> Result<()> {
        let mut schema_guard = self.schema.write().await;
        *schema_guard = schema;
        Ok(())
    }

    /// Add a query template
    pub async fn add_template(&self, template: QueryTemplate) -> Result<()> {
        let mut templates = self.templates.write().await;
        templates.push(template);
        Ok(())
    }

    /// Generate GraphQL query from natural language
    pub async fn generate(&self, nl_query: NaturalLanguageQuery) -> Result<GeneratedQuery> {
        // Step 1: Classify intent
        let intent_classifier = self.intent_classifier.read().await;
        let (intent, intent_confidence) = intent_classifier.classify(&nl_query.text);

        if intent_confidence < nl_query.confidence_threshold {
            return Err(anyhow!(
                "Low confidence in intent classification: {}",
                intent_confidence
            ));
        }

        // Step 2: Extract entities
        let entity_extractor = self.entity_extractor.read().await;
        let schema = self.schema.read().await;
        let entities = entity_extractor.extract(&nl_query.text, &schema);

        // Step 3: Find matching template
        let templates = self.templates.read().await;
        let matching_template = templates
            .iter()
            .find(|t| t.intent == intent)
            .ok_or_else(|| anyhow!("No template found for intent: {}", intent))?;

        // Step 4: Generate query from template
        let query = self.fill_template(matching_template, &entities).await?;

        // Step 5: Generate alternatives
        let alternatives = self.generate_alternatives(&intent, &entities).await?;

        // Step 6: Extract fields from entities
        let suggested_fields: Vec<String> = entities.keys().cloned().collect();

        let metadata = QueryMetadata {
            intent: intent.clone(),
            entities,
            operations: vec![intent.clone()],
            suggested_fields,
        };

        Ok(GeneratedQuery {
            query,
            confidence: intent_confidence,
            alternatives,
            metadata,
        })
    }

    /// Fill template with extracted entities
    async fn fill_template(
        &self,
        template: &QueryTemplate,
        entities: &HashMap<String, String>,
    ) -> Result<String> {
        let mut query = template.template.clone();

        // Replace placeholders with entity values
        for (key, value) in entities {
            let placeholder = format!("{{{}}}", key);
            query = query.replace(&placeholder, value);
        }

        Ok(query)
    }

    /// Generate alternative query interpretations
    async fn generate_alternatives(
        &self,
        intent: &str,
        entities: &HashMap<String, String>,
    ) -> Result<Vec<AlternativeQuery>> {
        let mut alternatives = Vec::new();

        // Generate alternative with different field selection
        if !entities.is_empty() {
            let alt_query = format!(
                "query {{ {}(filter: {}) {{ id }} }}",
                intent,
                self.format_filter(entities)
            );

            alternatives.push(AlternativeQuery {
                query: alt_query,
                confidence: 0.6,
                explanation: "Alternative with minimal field selection".to_string(),
            });
        }

        Ok(alternatives)
    }

    /// Format entities as GraphQL filter
    fn format_filter(&self, entities: &HashMap<String, String>) -> String {
        let filters: Vec<String> = entities
            .iter()
            .map(|(k, v)| format!("{}: \"{}\"", k, v))
            .collect();
        format!("{{ {} }}", filters.join(", "))
    }

    /// Get schema information
    pub async fn get_schema(&self) -> SchemaInfo {
        let schema = self.schema.read().await;
        schema.clone()
    }

    /// Train intent classifier with examples (placeholder for future)
    pub async fn train_intent_classifier(&self, _examples: Vec<(String, String)>) -> Result<()> {
        // In production, this would train a neural network
        // For now, we use the pre-initialized keyword-based classifier
        Ok(())
    }
}

impl Default for NaturalLanguageQueryGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_language_query_creation() {
        let query = NaturalLanguageQuery::new("find all users".to_string());
        assert_eq!(query.text, "find all users");
        assert_eq!(query.confidence_threshold, 0.7);
    }

    #[test]
    fn test_natural_language_query_with_context() {
        let query = NaturalLanguageQuery::new("get items".to_string())
            .with_context("e-commerce".to_string());
        assert_eq!(query.context, Some("e-commerce".to_string()));
    }

    #[test]
    fn test_schema_info_creation() {
        let mut schema = SchemaInfo::new();
        schema.add_type(
            "User".to_string(),
            vec!["id".to_string(), "name".to_string()],
        );
        assert_eq!(schema.types.len(), 1);
        assert_eq!(schema.fields.get("User").unwrap().len(), 2);
    }

    #[test]
    fn test_intent_classifier() {
        let classifier = IntentClassifier::new();
        let (intent, confidence) = classifier.classify("search for users");
        assert_eq!(intent, "search");
        assert!(confidence > 0.5);
    }

    #[test]
    fn test_intent_classifier_filter() {
        let classifier = IntentClassifier::new();
        let (intent, _) = classifier.classify("filter by name");
        assert_eq!(intent, "filter");
    }

    #[test]
    fn test_entity_extractor() {
        let extractor = EntityExtractor::new();
        let mut schema = SchemaInfo::new();
        schema.add_type(
            "User".to_string(),
            vec!["name".to_string(), "email".to_string()],
        );

        let entities = extractor.extract("get user name", &schema);
        assert!(entities.contains_key("name"));
    }

    #[tokio::test]
    async fn test_generator_creation() {
        let generator = NaturalLanguageQueryGenerator::new();
        let schema = generator.get_schema().await;
        assert_eq!(schema.types.len(), 0);
    }

    #[tokio::test]
    async fn test_register_schema() {
        let generator = NaturalLanguageQueryGenerator::new();
        let mut schema = SchemaInfo::new();
        schema.add_type("User".to_string(), vec!["id".to_string()]);

        generator.register_schema(schema).await.unwrap();
        let registered = generator.get_schema().await;
        assert_eq!(registered.types.len(), 1);
    }

    #[tokio::test]
    async fn test_add_template() {
        let generator = NaturalLanguageQueryGenerator::new();
        let template = QueryTemplate {
            name: "search_users".to_string(),
            intent: "search".to_string(),
            template: "query { users { id name } }".to_string(),
            required_entities: vec![],
        };

        generator.add_template(template).await.unwrap();
    }

    #[tokio::test]
    async fn test_generate_query() {
        let generator = NaturalLanguageQueryGenerator::new();

        // Setup schema
        let mut schema = SchemaInfo::new();
        schema.add_type(
            "User".to_string(),
            vec!["id".to_string(), "name".to_string()],
        );
        generator.register_schema(schema).await.unwrap();

        // Add template
        let template = QueryTemplate {
            name: "search_users".to_string(),
            intent: "search".to_string(),
            template: "query { users { id name } }".to_string(),
            required_entities: vec![],
        };
        generator.add_template(template).await.unwrap();

        // Generate query
        let nl_query = NaturalLanguageQuery::new("search for users".to_string());
        let result = generator.generate(nl_query).await;

        assert!(result.is_ok());
        let generated = result.unwrap();
        assert!(generated.query.contains("users"));
        assert_eq!(generated.metadata.intent, "search");
    }

    #[tokio::test]
    async fn test_generate_query_low_confidence() {
        let generator = NaturalLanguageQueryGenerator::new();
        let nl_query =
            NaturalLanguageQuery::new("xyzabc".to_string()).with_confidence_threshold(0.95);

        let result = generator.generate(nl_query).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_format_filter() {
        let generator = NaturalLanguageQueryGenerator::new();
        let mut entities = HashMap::new();
        entities.insert("name".to_string(), "John".to_string());
        entities.insert("age".to_string(), "30".to_string());

        let filter = generator.format_filter(&entities);
        assert!(filter.contains("name"));
        assert!(filter.contains("John"));
    }

    #[tokio::test]
    async fn test_train_intent_classifier() {
        let generator = NaturalLanguageQueryGenerator::new();
        let examples = vec![
            ("find all users".to_string(), "search".to_string()),
            ("filter by name".to_string(), "filter".to_string()),
        ];

        let result = generator.train_intent_classifier(examples).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_query_metadata_default() {
        let metadata = QueryMetadata::default();
        assert_eq!(metadata.intent, "unknown");
        assert!(metadata.entities.is_empty());
    }

    #[test]
    fn test_alternative_query() {
        let alt = AlternativeQuery {
            query: "query { users { id } }".to_string(),
            confidence: 0.7,
            explanation: "test".to_string(),
        };
        assert_eq!(alt.confidence, 0.7);
    }
}
