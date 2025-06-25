//! Base functionality shared across embedding models

use crate::{ModelConfig, ModelStats, TrainingStats, EmbeddingError, Triple, Vector};
use anyhow::Result;
use rand::prelude::SliceRandom;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use ndarray::{Array1, Array2};
use chrono::{DateTime, Utc};

/// Core data structures and functionality shared by all embedding models
#[derive(Debug)]
pub struct BaseModel {
    /// Model configuration
    pub config: ModelConfig,
    /// Unique model identifier
    pub model_id: Uuid,
    /// Entity to index mapping
    pub entity_to_id: HashMap<String, usize>,
    /// Index to entity mapping
    pub id_to_entity: HashMap<usize, String>,
    /// Relation to index mapping
    pub relation_to_id: HashMap<String, usize>,
    /// Index to relation mapping
    pub id_to_relation: HashMap<usize, String>,
    /// Training triples (subject_id, predicate_id, object_id)
    pub triples: Vec<(usize, usize, usize)>,
    /// Set of all positive triples for fast lookup
    pub positive_triples: HashSet<(usize, usize, usize)>,
    /// Whether the model has been trained
    pub is_trained: bool,
    /// Model creation time
    pub creation_time: DateTime<Utc>,
    /// Last training time
    pub last_training_time: Option<DateTime<Utc>>,
}

impl BaseModel {
    /// Create a new base model
    pub fn new(config: ModelConfig) -> Self {
        Self {
            model_id: Uuid::new_v4(),
            config,
            entity_to_id: HashMap::new(),
            id_to_entity: HashMap::new(),
            relation_to_id: HashMap::new(),
            id_to_relation: HashMap::new(),
            triples: Vec::new(),
            positive_triples: HashSet::new(),
            is_trained: false,
            creation_time: Utc::now(),
            last_training_time: None,
        }
    }
    
    /// Add a triple to the model
    pub fn add_triple(&mut self, triple: Triple) -> Result<()> {
        let subject_str = triple.subject.to_string();
        let predicate_str = triple.predicate.to_string();
        let object_str = triple.object.to_string();
        
        // Get or create entity IDs
        let subject_id = self.get_or_create_entity_id(subject_str);
        let object_id = self.get_or_create_entity_id(object_str);
        
        // Get or create relation ID
        let predicate_id = self.get_or_create_relation_id(predicate_str);
        
        // Add triple
        let triple_ids = (subject_id, predicate_id, object_id);
        if !self.positive_triples.contains(&triple_ids) {
            self.triples.push(triple_ids);
            self.positive_triples.insert(triple_ids);
        }
        
        Ok(())
    }
    
    /// Get or create entity ID
    fn get_or_create_entity_id(&mut self, entity: String) -> usize {
        if let Some(&id) = self.entity_to_id.get(&entity) {
            id
        } else {
            let id = self.entity_to_id.len();
            self.entity_to_id.insert(entity.clone(), id);
            self.id_to_entity.insert(id, entity);
            id
        }
    }
    
    /// Get or create relation ID
    fn get_or_create_relation_id(&mut self, relation: String) -> usize {
        if let Some(&id) = self.relation_to_id.get(&relation) {
            id
        } else {
            let id = self.relation_to_id.len();
            self.relation_to_id.insert(relation.clone(), id);
            self.id_to_relation.insert(id, relation);
            id
        }
    }
    
    /// Get entity ID
    pub fn get_entity_id(&self, entity: &str) -> Option<usize> {
        self.entity_to_id.get(entity).copied()
    }
    
    /// Get relation ID
    pub fn get_relation_id(&self, relation: &str) -> Option<usize> {
        self.relation_to_id.get(relation).copied()
    }
    
    /// Get entity string from ID
    pub fn get_entity(&self, id: usize) -> Option<&String> {
        self.id_to_entity.get(&id)
    }
    
    /// Get relation string from ID
    pub fn get_relation(&self, id: usize) -> Option<&String> {
        self.id_to_relation.get(&id)
    }
    
    /// Get number of entities
    pub fn num_entities(&self) -> usize {
        self.entity_to_id.len()
    }
    
    /// Get number of relations
    pub fn num_relations(&self) -> usize {
        self.relation_to_id.len()
    }
    
    /// Get number of triples
    pub fn num_triples(&self) -> usize {
        self.triples.len()
    }
    
    /// Get all entity strings
    pub fn get_entities(&self) -> Vec<String> {
        self.entity_to_id.keys().cloned().collect()
    }
    
    /// Get all relation strings
    pub fn get_relations(&self) -> Vec<String> {
        self.relation_to_id.keys().cloned().collect()
    }
    
    /// Check if a triple exists in the knowledge base
    pub fn has_triple(&self, subject_id: usize, predicate_id: usize, object_id: usize) -> bool {
        self.positive_triples.contains(&(subject_id, predicate_id, object_id))
    }
    
    /// Generate negative samples for training
    pub fn generate_negative_samples(&self, num_samples: usize, rng: &mut impl rand::Rng) -> Vec<(usize, usize, usize)> {
        let mut negative_samples = Vec::new();
        let num_entities = self.num_entities();
        
        while negative_samples.len() < num_samples {
            // Choose a random positive triple
            if let Some(&(s, p, o)) = self.triples.choose(rng) {
                // Corrupt either subject or object
                let corrupt_subject = rng.gen_bool(0.5);
                
                let negative_triple = if corrupt_subject {
                    let new_subject = rng.gen_range(0..num_entities);
                    (new_subject, p, o)
                } else {
                    let new_object = rng.gen_range(0..num_entities);
                    (s, p, new_object)
                };
                
                // Make sure it's actually negative
                if !self.has_triple(negative_triple.0, negative_triple.1, negative_triple.2) {
                    negative_samples.push(negative_triple);
                }
            }
        }
        
        negative_samples
    }
    
    /// Get model statistics
    pub fn get_stats(&self, model_type: &str) -> ModelStats {
        ModelStats {
            num_entities: self.num_entities(),
            num_relations: self.num_relations(),
            num_triples: self.num_triples(),
            dimensions: self.config.dimensions,
            is_trained: self.is_trained,
            model_type: model_type.to_string(),
            creation_time: self.creation_time,
            last_training_time: self.last_training_time,
        }
    }
    
    /// Clear all data
    pub fn clear(&mut self) {
        self.entity_to_id.clear();
        self.id_to_entity.clear();
        self.relation_to_id.clear();
        self.id_to_relation.clear();
        self.triples.clear();
        self.positive_triples.clear();
        self.is_trained = false;
        self.last_training_time = None;
    }
    
    /// Mark model as trained
    pub fn mark_trained(&mut self) {
        self.is_trained = true;
        self.last_training_time = Some(Utc::now());
    }
}