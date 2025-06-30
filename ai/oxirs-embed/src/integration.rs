//! Integration utilities with other OxiRS components

use crate::{EmbeddingModel, Vector};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Integration bridge between oxirs-embed and vector stores
pub struct VectorStoreBridge {
    entity_mappings: HashMap<String, String>,
    relation_mappings: HashMap<String, String>,
    prefix_config: PrefixConfig,
}

/// Configuration for URI prefixes in vector store
#[derive(Debug, Clone)]
pub struct PrefixConfig {
    pub entity_prefix: String,
    pub relation_prefix: String,
    pub use_namespaces: bool,
}

impl Default for PrefixConfig {
    fn default() -> Self {
        Self {
            entity_prefix: "kg:entity:".to_string(),
            relation_prefix: "kg:relation:".to_string(),
            use_namespaces: true,
        }
    }
}

impl VectorStoreBridge {
    /// Create a new bridge
    pub fn new() -> Self {
        Self {
            entity_mappings: HashMap::new(),
            relation_mappings: HashMap::new(),
            prefix_config: PrefixConfig::default(),
        }
    }

    /// Create bridge with custom prefix config
    pub fn with_prefix_config(prefix_config: PrefixConfig) -> Self {
        Self {
            entity_mappings: HashMap::new(),
            relation_mappings: HashMap::new(),
            prefix_config,
        }
    }

    /// Sync all embeddings from a model to the vector store
    pub fn sync_model_embeddings(&mut self, model: &dyn EmbeddingModel) -> Result<SyncStats> {
        let start_time = std::time::Instant::now();
        let mut sync_stats = SyncStats::default();

        info!("Starting embedding synchronization to vector store");

        // Sync entity embeddings
        let entities = model.get_entities();
        for entity in &entities {
            match model.get_entity_embedding(entity) {
                Ok(_embedding) => {
                    let uri = self.generate_entity_uri(entity);
                    self.entity_mappings.insert(entity.clone(), uri);
                    sync_stats.entities_synced += 1;
                }
                Err(e) => {
                    warn!("Failed to get embedding for entity {}: {}", entity, e);
                    sync_stats.errors.push(format!("Entity {}: {}", entity, e));
                }
            }
        }

        // Sync relation embeddings
        let relations = model.get_relations();
        for relation in &relations {
            match model.get_relation_embedding(relation) {
                Ok(_embedding) => {
                    let uri = self.generate_relation_uri(relation);
                    self.relation_mappings.insert(relation.clone(), uri);
                    sync_stats.relations_synced += 1;
                }
                Err(e) => {
                    warn!("Failed to get embedding for relation {}: {}", relation, e);
                    sync_stats
                        .errors
                        .push(format!("Relation {}: {}", relation, e));
                }
            }
        }

        sync_stats.sync_duration = start_time.elapsed();
        info!(
            "Embedding sync completed: {} entities, {} relations, {} errors",
            sync_stats.entities_synced,
            sync_stats.relations_synced,
            sync_stats.errors.len()
        );

        Ok(sync_stats)
    }

    /// Find similar entities using vector similarity
    pub fn find_similar_entities(&self, entity: &str, _k: usize) -> Result<Vec<(String, f32)>> {
        if let Some(_uri) = self.entity_mappings.get(entity) {
            // This would require extending VectorStore to support querying by URI
            // For now, we return empty results
            debug!("Searching for entities similar to: {}", entity);
            Ok(vec![])
        } else {
            Err(anyhow!("Entity not found in mappings: {}", entity))
        }
    }

    /// Find similar relations using vector similarity
    pub fn find_similar_relations(&self, relation: &str, _k: usize) -> Result<Vec<(String, f32)>> {
        if let Some(_uri) = self.relation_mappings.get(relation) {
            debug!("Searching for relations similar to: {}", relation);
            Ok(vec![])
        } else {
            Err(anyhow!("Relation not found in mappings: {}", relation))
        }
    }

    /// Generate URI for entity
    fn generate_entity_uri(&self, entity: &str) -> String {
        if self.prefix_config.use_namespaces {
            format!("{}{}", self.prefix_config.entity_prefix, entity)
        } else {
            entity.to_string()
        }
    }

    /// Generate URI for relation
    fn generate_relation_uri(&self, relation: &str) -> String {
        if self.prefix_config.use_namespaces {
            format!("{}{}", self.prefix_config.relation_prefix, relation)
        } else {
            relation.to_string()
        }
    }

    /// Get sync statistics
    pub fn get_sync_info(&self) -> SyncInfo {
        SyncInfo {
            entities_mapped: self.entity_mappings.len(),
            relations_mapped: self.relation_mappings.len(),
            vector_store_stats: None,
        }
    }

    /// Clear all mappings
    pub fn clear_mappings(&mut self) {
        self.entity_mappings.clear();
        self.relation_mappings.clear();
        info!("Cleared all entity and relation mappings");
    }
}

impl Default for VectorStoreBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from synchronization operation
#[derive(Debug, Clone, Default)]
pub struct SyncStats {
    pub entities_synced: usize,
    pub relations_synced: usize,
    pub errors: Vec<String>,
    pub sync_duration: std::time::Duration,
}

/// Information about current sync state
#[derive(Debug, Clone)]
pub struct SyncInfo {
    pub entities_mapped: usize,
    pub relations_mapped: usize,
    pub vector_store_stats: Option<(usize, usize)>,
}

/// Integration with oxirs-chat for conversational AI
pub struct ChatIntegration {
    model: Box<dyn EmbeddingModel>,
    context_window: usize,
    similarity_threshold: f32,
    personalization: PersonalizationEngine,
    multilingual: MultilingualSupport,
}

impl ChatIntegration {
    /// Create new chat integration
    pub fn new(model: Box<dyn EmbeddingModel>) -> Self {
        Self {
            model,
            context_window: 10,
            similarity_threshold: 0.7,
            personalization: PersonalizationEngine::new(),
            multilingual: MultilingualSupport::new(),
        }
    }

    /// Configure context window size
    pub fn with_context_window(mut self, window_size: usize) -> Self {
        self.context_window = window_size;
        self
    }

    /// Configure similarity threshold for relevant entities
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Extract relevant entities from a query
    pub fn extract_relevant_entities(&self, query: &str) -> Result<Vec<String>> {
        // This is a simplified implementation
        // In practice, this would use NLP techniques to identify entities
        let entities = self.model.get_entities();
        let mut relevant = Vec::new();

        for entity in entities {
            // Simple substring matching - would be replaced with proper NLP
            if query.to_lowercase().contains(&entity.to_lowercase()) {
                relevant.push(entity);
            }
        }

        Ok(relevant)
    }

    /// Generate context embeddings for a conversation
    pub fn generate_context_embedding(&self, messages: &[String]) -> Result<Vector> {
        if messages.is_empty() {
            return Err(anyhow!("No messages provided"));
        }

        // Take the last N messages based on context window
        let _recent_messages: Vec<&String> =
            messages.iter().rev().take(self.context_window).collect();

        // For now, just return a dummy embedding
        // In practice, this would combine message embeddings intelligently
        let dummy_values = vec![0.0; 100]; // Would be model's dimension
        Ok(Vector::new(
            dummy_values.into_iter().map(|x| x as f32).collect(),
        ))
    }

    /// Generate personalized embeddings for a user
    pub async fn generate_personalized_embedding(
        &mut self,
        user_id: &str,
        query: &str,
        conversation_history: &[String],
    ) -> Result<Vector> {
        // Get user profile and preferences
        let user_profile = self.personalization.get_user_profile(user_id)?.clone();

        // Apply user preferences to query embedding
        let embeddings = self.model.encode(&[query.to_string()]).await?;
        let base_embedding = Vector::new(embeddings[0].clone());
        let personalized_embedding = self.personalization.apply_user_preferences(
            &base_embedding,
            &user_profile,
            conversation_history,
        )?;

        Ok(personalized_embedding)
    }

    /// Update user profile based on interaction
    pub fn update_user_profile(
        &mut self,
        user_id: &str,
        query: &str,
        response_feedback: Option<f32>,
        interaction_type: InteractionType,
    ) -> Result<()> {
        self.personalization.update_user_profile(
            user_id,
            query,
            response_feedback,
            interaction_type,
        )
    }

    /// Translate query to target language
    pub async fn translate_query(
        &self,
        query: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<String> {
        self.multilingual
            .translate_text(query, source_lang, target_lang)
            .await
    }

    /// Detect language of input text
    pub async fn detect_language(&self, text: &str) -> Result<LanguageDetection> {
        self.multilingual.detect_language(text).await
    }

    /// Generate cross-lingual embeddings
    pub async fn generate_cross_lingual_embedding(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<Vector> {
        self.multilingual
            .generate_cross_lingual_embedding(text, source_lang, target_lang, &*self.model)
            .await
    }

    /// Get multilingual entity alignment
    pub async fn align_entities_across_languages(
        &self,
        entity: &str,
        source_lang: &str,
        target_langs: &[String],
    ) -> Result<HashMap<String, String>> {
        self.multilingual
            .align_entities(entity, source_lang, target_langs)
            .await
    }
}

/// SPARQL integration for query enhancement
pub struct SparqlIntegration {
    #[allow(dead_code)]
    model: Box<dyn EmbeddingModel>,
    #[allow(dead_code)]
    similarity_boost: f32,
}

impl SparqlIntegration {
    /// Create new SPARQL integration
    pub fn new(model: Box<dyn EmbeddingModel>) -> Self {
        Self {
            model,
            similarity_boost: 0.1,
        }
    }

    /// Enhance SPARQL query with similarity-based suggestions
    pub fn enhance_query(&self, sparql_query: &str) -> Result<EnhancedQuery> {
        // Parse basic patterns from SPARQL (simplified)
        let entities = self.extract_entities_from_sparql(sparql_query)?;
        let relations = self.extract_relations_from_sparql(sparql_query)?;

        let mut suggestions = Vec::new();

        // Find similar entities
        for entity in &entities {
            // This would use actual similarity computation
            suggestions.push(QuerySuggestion {
                suggestion_type: SuggestionType::SimilarEntity,
                original: entity.clone(),
                suggested: format!("similar_to_{}", entity),
                confidence: 0.8,
            });
        }

        // Find similar relations
        for relation in &relations {
            suggestions.push(QuerySuggestion {
                suggestion_type: SuggestionType::SimilarRelation,
                original: relation.clone(),
                suggested: format!("similar_to_{}", relation),
                confidence: 0.7,
            });
        }

        Ok(EnhancedQuery {
            original_query: sparql_query.to_string(),
            entities_found: entities,
            relations_found: relations,
            suggestions,
        })
    }

    /// Extract entities from SPARQL query (simplified)
    fn extract_entities_from_sparql(&self, query: &str) -> Result<Vec<String>> {
        // This is a very simplified extraction
        // A real implementation would use a proper SPARQL parser
        let mut entities = Vec::new();

        for line in query.lines() {
            if line.contains("http://") {
                // Extract URIs that might be entities
                if let Some(start) = line.find("http://") {
                    if let Some(end) = line[start..].find(' ') {
                        let uri = &line[start..start + end];
                        entities.push(uri.to_string());
                    }
                }
            }
        }

        Ok(entities)
    }

    /// Extract relations from SPARQL query (simplified)
    fn extract_relations_from_sparql(&self, query: &str) -> Result<Vec<String>> {
        // Simplified relation extraction
        let mut relations = Vec::new();

        for line in query.lines() {
            if line.contains("?") && line.contains("http://") {
                // Look for patterns like "?s <relation> ?o"
                if let Some(start) = line.find('<') {
                    if let Some(end) = line.find('>') {
                        let relation = &line[start + 1..end];
                        relations.push(relation.to_string());
                    }
                }
            }
        }

        Ok(relations)
    }
}

/// Enhanced SPARQL query with suggestions
#[derive(Debug, Clone)]
pub struct EnhancedQuery {
    pub original_query: String,
    pub entities_found: Vec<String>,
    pub relations_found: Vec<String>,
    pub suggestions: Vec<QuerySuggestion>,
}

/// Query enhancement suggestion
#[derive(Debug, Clone)]
pub struct QuerySuggestion {
    pub suggestion_type: SuggestionType,
    pub original: String,
    pub suggested: String,
    pub confidence: f32,
}

/// Types of query suggestions
#[derive(Debug, Clone)]
pub enum SuggestionType {
    SimilarEntity,
    SimilarRelation,
    AlternativePattern,
    ExpansionSuggestion,
}

/// Personalization engine for user-specific embeddings
pub struct PersonalizationEngine {
    user_profiles: HashMap<String, UserProfile>,
    interaction_history: HashMap<String, Vec<UserInteraction>>,
    preference_weights: PreferenceWeights,
}

impl PersonalizationEngine {
    pub fn new() -> Self {
        Self {
            user_profiles: HashMap::new(),
            interaction_history: HashMap::new(),
            preference_weights: PreferenceWeights::default(),
        }
    }

    /// Get or create user profile
    pub fn get_user_profile(&mut self, user_id: &str) -> Result<&UserProfile> {
        if !self.user_profiles.contains_key(user_id) {
            let profile = UserProfile::new(user_id.to_string());
            self.user_profiles.insert(user_id.to_string(), profile);
        }

        self.user_profiles
            .get(user_id)
            .ok_or_else(|| anyhow!("Failed to get user profile for {}", user_id))
    }

    /// Apply user preferences to embedding
    pub fn apply_user_preferences(
        &self,
        base_embedding: &Vector,
        user_profile: &UserProfile,
        conversation_history: &[String],
    ) -> Result<Vector> {
        let mut personalized = base_embedding.clone();

        // Apply domain preferences
        for (domain, weight) in &user_profile.domain_preferences {
            if conversation_history.iter().any(|msg| msg.contains(domain)) {
                // Boost embedding components related to preferred domains
                for i in 0..personalized.values.len() {
                    personalized.values[i] *= 1.0 + (weight * self.preference_weights.domain_boost);
                }
            }
        }

        // Apply recent interaction patterns
        let recent_interactions = self.get_recent_interactions(&user_profile.user_id, 10);
        if !recent_interactions.is_empty() {
            let avg_sentiment = recent_interactions
                .iter()
                .map(|i| i.sentiment_score.unwrap_or(0.0))
                .sum::<f32>()
                / recent_interactions.len() as f32;

            // Adjust embedding based on user's typical sentiment
            for i in 0..personalized.values.len() {
                personalized.values[i] *=
                    1.0 + (avg_sentiment * self.preference_weights.sentiment_influence);
            }
        }

        Ok(personalized)
    }

    /// Update user profile based on interaction
    pub fn update_user_profile(
        &mut self,
        user_id: &str,
        query: &str,
        response_feedback: Option<f32>,
        interaction_type: InteractionType,
    ) -> Result<()> {
        let interaction = UserInteraction {
            timestamp: chrono::Utc::now(),
            query: query.to_string(),
            interaction_type,
            response_feedback,
            sentiment_score: self.analyze_query_sentiment(query),
        };

        // Add to interaction history
        self.interaction_history
            .entry(user_id.to_string())
            .or_insert_with(Vec::new)
            .push(interaction.clone());

        // Update user profile
        if let Some(profile) = self.user_profiles.get_mut(user_id) {
            profile.update_from_interaction(&interaction);
        }

        Ok(())
    }

    /// Get recent interactions for a user
    fn get_recent_interactions(&self, user_id: &str, limit: usize) -> Vec<&UserInteraction> {
        self.interaction_history
            .get(user_id)
            .map(|history| history.iter().rev().take(limit).collect())
            .unwrap_or_default()
    }

    /// Simple sentiment analysis for query
    fn analyze_query_sentiment(&self, query: &str) -> Option<f32> {
        let positive_words = ["good", "great", "excellent", "amazing", "wonderful"];
        let negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"];

        let query_lower = query.to_lowercase();
        let positive_count = positive_words
            .iter()
            .filter(|&&word| query_lower.contains(word))
            .count();
        let negative_count = negative_words
            .iter()
            .filter(|&&word| query_lower.contains(word))
            .count();

        if positive_count + negative_count == 0 {
            return None;
        }

        let sentiment = (positive_count as f32 - negative_count as f32)
            / (positive_count + negative_count) as f32;
        Some(sentiment)
    }
}

/// User profile for personalization
#[derive(Debug, Clone)]
pub struct UserProfile {
    pub user_id: String,
    pub domain_preferences: HashMap<String, f32>,
    pub entity_preferences: HashMap<String, f32>,
    pub interaction_patterns: InteractionPatterns,
    pub language_preferences: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl UserProfile {
    pub fn new(user_id: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            user_id,
            domain_preferences: HashMap::new(),
            entity_preferences: HashMap::new(),
            interaction_patterns: InteractionPatterns::default(),
            language_preferences: vec!["en".to_string()],
            created_at: now,
            last_updated: now,
        }
    }

    /// Update profile based on user interaction
    pub fn update_from_interaction(&mut self, interaction: &UserInteraction) {
        self.last_updated = chrono::Utc::now();

        // Update interaction patterns
        self.interaction_patterns.total_interactions += 1;
        match interaction.interaction_type {
            InteractionType::Query => self.interaction_patterns.query_count += 1,
            InteractionType::Feedback => self.interaction_patterns.feedback_count += 1,
            InteractionType::EntityLookup => self.interaction_patterns.entity_lookup_count += 1,
        }

        // Update average sentiment
        if let Some(sentiment) = interaction.sentiment_score {
            let current_avg = self.interaction_patterns.average_sentiment;
            let total = self.interaction_patterns.total_interactions as f32;
            self.interaction_patterns.average_sentiment =
                (current_avg * (total - 1.0) + sentiment) / total;
        }

        // Extract and update domain preferences from query
        self.extract_domain_preferences(&interaction.query);
    }

    /// Extract domain preferences from query text
    fn extract_domain_preferences(&mut self, query: &str) {
        let domains = [
            "science",
            "technology",
            "medicine",
            "business",
            "education",
            "sports",
            "entertainment",
            "politics",
            "history",
            "art",
        ];

        for domain in &domains {
            if query.to_lowercase().contains(domain) {
                let current = self
                    .domain_preferences
                    .get(&domain.to_string())
                    .copied()
                    .unwrap_or(0.0);
                self.domain_preferences
                    .insert(domain.to_string(), current + 0.1);
            }
        }
    }
}

/// User interaction patterns
#[derive(Debug, Clone, Default)]
pub struct InteractionPatterns {
    pub total_interactions: u32,
    pub query_count: u32,
    pub feedback_count: u32,
    pub entity_lookup_count: u32,
    pub average_sentiment: f32,
    pub preferred_response_length: Option<usize>,
}

/// Types of user interactions
#[derive(Debug, Clone)]
pub enum InteractionType {
    Query,
    Feedback,
    EntityLookup,
}

/// User interaction record
#[derive(Debug, Clone)]
pub struct UserInteraction {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub query: String,
    pub interaction_type: InteractionType,
    pub response_feedback: Option<f32>,
    pub sentiment_score: Option<f32>,
}

/// Weights for preference application
#[derive(Debug, Clone)]
pub struct PreferenceWeights {
    pub domain_boost: f32,
    pub entity_boost: f32,
    pub sentiment_influence: f32,
    pub recency_decay: f32,
}

impl Default for PreferenceWeights {
    fn default() -> Self {
        Self {
            domain_boost: 0.1,
            entity_boost: 0.15,
            sentiment_influence: 0.05,
            recency_decay: 0.95,
        }
    }
}

/// Multilingual support for chat integration
pub struct MultilingualSupport {
    supported_languages: Vec<String>,
    translation_cache: HashMap<String, String>,
    language_models: HashMap<String, LanguageModel>,
}

impl MultilingualSupport {
    pub fn new() -> Self {
        Self {
            supported_languages: vec![
                "en".to_string(),
                "es".to_string(),
                "fr".to_string(),
                "de".to_string(),
                "it".to_string(),
                "pt".to_string(),
                "zh".to_string(),
                "ja".to_string(),
                "ko".to_string(),
                "ar".to_string(),
                "hi".to_string(),
                "ru".to_string(),
            ],
            translation_cache: HashMap::new(),
            language_models: HashMap::new(),
        }
    }

    /// Translate text between languages
    pub async fn translate_text(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<String> {
        if source_lang == target_lang {
            return Ok(text.to_string());
        }

        let cache_key = format!("{}:{}:{}", source_lang, target_lang, text);
        if let Some(cached) = self.translation_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Mock translation implementation
        // In practice, this would call a translation service
        let translated = match target_lang {
            "es" => format!("[ES] {}", text),
            "fr" => format!("[FR] {}", text),
            "de" => format!("[DE] {}", text),
            "zh" => format!("[ZH] {}", text),
            _ => format!("[{}] {}", target_lang.to_uppercase(), text),
        };

        Ok(translated)
    }

    /// Detect language of input text
    pub async fn detect_language(&self, text: &str) -> Result<LanguageDetection> {
        // Simple language detection based on common words
        let text_lower = text.to_lowercase();

        let mut scores = HashMap::new();

        // English indicators
        let en_words = ["the", "and", "is", "hello", "world", "of", "to", "in"];
        let en_score = en_words
            .iter()
            .filter(|&&word| text_lower.contains(word))
            .count();
        scores.insert("en", en_score);

        // Spanish indicators
        let es_words = ["el", "y", "es", "hola", "buenos", "dias", "de", "en", "la"];
        let es_score = es_words
            .iter()
            .filter(|&&word| text_lower.contains(word))
            .count();
        scores.insert("es", es_score);

        // French indicators
        let fr_words = ["le", "et", "est", "bonjour", "de", "la", "les"];
        let fr_score = fr_words
            .iter()
            .filter(|&&word| text_lower.contains(word))
            .count();
        scores.insert("fr", fr_score);

        // German indicators
        let de_words = ["der", "und", "ist", "hallo", "von", "die", "das"];
        let de_score = de_words
            .iter()
            .filter(|&&word| text_lower.contains(word))
            .count();
        scores.insert("de", de_score);

        // Find language with highest score
        let detected_lang = scores
            .iter()
            .max_by_key(|(_, &score)| score)
            .map(|(lang, _)| *lang)
            .unwrap_or("en");

        Ok(LanguageDetection {
            language_code: detected_lang.to_string(),
            confidence: 0.85,
            alternatives: vec![
                ("en".to_string(), 0.7),
                ("es".to_string(), 0.2),
                ("fr".to_string(), 0.1),
            ],
        })
    }

    /// Generate cross-lingual embeddings
    pub async fn generate_cross_lingual_embedding(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
        model: &dyn EmbeddingModel,
    ) -> Result<Vector> {
        // For cross-lingual embeddings, we would typically:
        // 1. Use a multilingual embedding model
        // 2. Or translate text and generate embedding
        // 3. Or use language-specific models with alignment

        let translated_text = self.translate_text(text, source_lang, target_lang).await?;
        let embeddings = model.encode(&[translated_text]).await?;
        Ok(Vector::new(embeddings[0].clone()))
    }

    /// Align entities across languages
    pub async fn align_entities(
        &self,
        entity: &str,
        source_lang: &str,
        target_langs: &[String],
    ) -> Result<HashMap<String, String>> {
        let mut alignments = HashMap::new();

        for target_lang in target_langs {
            if target_lang == source_lang {
                alignments.insert(target_lang.clone(), entity.to_string());
                continue;
            }

            // Mock entity alignment - in practice would use knowledge bases
            let aligned_entity = match target_lang.as_str() {
                "es" => format!("{}_es", entity),
                "fr" => format!("{}_fr", entity),
                "de" => format!("{}_de", entity),
                "zh" => format!("{}_zh", entity),
                _ => format!("{}_{}", entity, target_lang),
            };

            alignments.insert(target_lang.clone(), aligned_entity);
        }

        Ok(alignments)
    }
}

/// Language detection result
#[derive(Debug, Clone)]
pub struct LanguageDetection {
    pub language_code: String,
    pub confidence: f32,
    pub alternatives: Vec<(String, f32)>,
}

/// Language model information
#[derive(Debug, Clone)]
pub struct LanguageModel {
    pub model_id: String,
    pub language_code: String,
    pub model_type: String,
    pub embedding_dimension: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TransE;
    use crate::ModelConfig;

    #[test]
    fn test_vector_store_bridge() {
        let config = ModelConfig::default().with_dimensions(10);
        let _model = TransE::new(config);

        let bridge = VectorStoreBridge::new();

        // Test URI generation
        let entity_uri = bridge.generate_entity_uri("test_entity");
        assert!(entity_uri.starts_with("kg:entity:"));

        let relation_uri = bridge.generate_relation_uri("test_relation");
        assert!(relation_uri.starts_with("kg:relation:"));
    }

    #[test]
    fn test_sparql_integration() -> Result<()> {
        let config = ModelConfig::default().with_dimensions(10);
        let model = TransE::new(config);

        let integration = SparqlIntegration::new(Box::new(model));

        let test_query = "SELECT ?s ?o WHERE { ?s <http://example.org/knows> ?o }";
        let enhanced = integration.enhance_query(test_query)?;

        assert_eq!(enhanced.original_query, test_query);
        assert!(!enhanced.suggestions.is_empty());

        Ok(())
    }

    #[test]
    fn test_personalization_engine() {
        let mut engine = PersonalizationEngine::new();
        let user_id = "test_user";

        // Test user profile creation
        let profile = engine.get_user_profile(user_id).unwrap();
        assert_eq!(profile.user_id, user_id);

        // Test interaction update
        engine
            .update_user_profile(
                user_id,
                "What is machine learning?",
                Some(0.9),
                InteractionType::Query,
            )
            .unwrap();

        let history = engine.get_recent_interactions(user_id, 5);
        assert_eq!(history.len(), 1);
    }

    #[tokio::test]
    async fn test_multilingual_support() -> Result<()> {
        let multilingual = MultilingualSupport::new();

        // Test language detection with English text
        let detection_en = multilingual.detect_language("Hello world").await?;
        assert_eq!(detection_en.language_code, "en");

        // Test language detection with Spanish text
        let detection_es = multilingual.detect_language("Hola y buenos dias").await?;
        assert_eq!(detection_es.language_code, "es");

        // Test translation
        let translated = multilingual
            .translate_text("Hello world", "en", "es")
            .await?;
        assert!(translated.contains("[ES]"));

        // Test entity alignment
        let alignments = multilingual
            .align_entities("person", "en", &["es".to_string(), "fr".to_string()])
            .await?;
        assert_eq!(alignments.len(), 2);

        Ok(())
    }
}
