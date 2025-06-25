//! Integration utilities with other OxiRS components

use crate::{EmbeddingModel, ModelStats};
use anyhow::{anyhow, Result};
use crate::Vector;
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
            prefix_config: PrefixConfig::default(),
        }
    }
    
    /// Configure URI prefixes
    pub fn with_prefix_config(mut self, config: PrefixConfig) -> Self {
        self.prefix_config = config;
        self
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
                Ok(embedding) => {
                    let uri = self.generate_entity_uri(entity);
                    match self.vector_store.index_vector(uri.clone(), embedding) {
                        Ok(_) => {
                            self.entity_mappings.insert(entity.clone(), uri);
                            sync_stats.entities_synced += 1;
                        }
                        Err(e) => {
                            warn!("Failed to index entity {}: {}", entity, e);
                            sync_stats.errors.push(format!("Entity {}: {}", entity, e));
                        }
                    }
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
                Ok(embedding) => {
                    let uri = self.generate_relation_uri(relation);
                    match self.vector_store.index_vector(uri.clone(), embedding) {
                        Ok(_) => {
                            self.relation_mappings.insert(relation.clone(), uri);
                            sync_stats.relations_synced += 1;
                        }
                        Err(e) => {
                            warn!("Failed to index relation {}: {}", relation, e);
                            sync_stats.errors.push(format!("Relation {}: {}", relation, e));
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to get embedding for relation {}: {}", relation, e);
                    sync_stats.errors.push(format!("Relation {}: {}", relation, e));
                }
            }
        }
        
        sync_stats.sync_duration = start_time.elapsed();
        info!("Embedding sync completed: {} entities, {} relations, {} errors", 
              sync_stats.entities_synced, sync_stats.relations_synced, sync_stats.errors.len());
        
        Ok(sync_stats)
    }
    
    /// Find similar entities using vector similarity
    pub fn find_similar_entities(&self, entity: &str, k: usize) -> Result<Vec<(String, f32)>> {
        if let Some(uri) = self.entity_mappings.get(entity) {
            // This would require extending VectorStore to support querying by URI
            // For now, we return empty results
            debug!("Searching for entities similar to: {}", entity);
            Ok(vec![])
        } else {
            Err(anyhow!("Entity not found in mappings: {}", entity))
        }
    }
    
    /// Find similar relations using vector similarity
    pub fn find_similar_relations(&self, relation: &str, k: usize) -> Result<Vec<(String, f32)>> {
        if let Some(uri) = self.relation_mappings.get(relation) {
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
            vector_store_stats: self.vector_store.embedding_stats(),
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
}

impl ChatIntegration {
    /// Create new chat integration
    pub fn new(model: Box<dyn EmbeddingModel>) -> Self {
        Self {
            model,
            context_window: 10,
            similarity_threshold: 0.7,
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
        let recent_messages: Vec<&String> = messages
            .iter()
            .rev()
            .take(self.context_window)
            .collect();
        
        // For now, just return a dummy embedding
        // In practice, this would combine message embeddings intelligently
        let dummy_values = vec![0.0; 100]; // Would be model's dimension
        Ok(Vector::new(dummy_values.into_iter().map(|x| x as f32).collect()))
    }
}

/// SPARQL integration for query enhancement
pub struct SparqlIntegration {
    model: Box<dyn EmbeddingModel>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TransE;
    use crate::ModelConfig;
    
    #[test]
    fn test_vector_store_bridge() {
        let config = ModelConfig::default().with_dimensions(10);
        let model = TransE::new(config);
        
        let mut bridge = VectorStoreBridge::new();
        
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
}