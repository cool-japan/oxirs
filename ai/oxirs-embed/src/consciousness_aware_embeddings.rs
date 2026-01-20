//! Consciousness-Aware Embedding System
//!
//! This module implements a revolutionary consciousness-aware embedding system that
//! can adapt based on context, reasoning, and multi-dimensional awareness patterns.
//! Inspired by theories of consciousness, cognitive science, and advanced AI research.

use crate::ModelConfig;
use anyhow::Result;
use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Levels of consciousness awareness in the embedding system
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConsciousnessLevel {
    /// Basic reactive responses
    Reactive,
    /// Simple pattern recognition
    Associative,
    /// Basic self-awareness
    SelfAware,
    /// Complex reasoning and planning
    Reflective,
    /// Meta-cognitive awareness
    MetaCognitive,
    /// Transcendent consciousness
    Transcendent,
}

impl ConsciousnessLevel {
    pub fn awareness_factor(&self) -> f32 {
        match self {
            ConsciousnessLevel::Reactive => 0.1,
            ConsciousnessLevel::Associative => 0.3,
            ConsciousnessLevel::SelfAware => 0.5,
            ConsciousnessLevel::Reflective => 0.7,
            ConsciousnessLevel::MetaCognitive => 0.9,
            ConsciousnessLevel::Transcendent => 1.0,
        }
    }

    pub fn complexity_threshold(&self) -> f32 {
        match self {
            ConsciousnessLevel::Reactive => 0.1,
            ConsciousnessLevel::Associative => 0.2,
            ConsciousnessLevel::SelfAware => 0.4,
            ConsciousnessLevel::Reflective => 0.6,
            ConsciousnessLevel::MetaCognitive => 0.8,
            ConsciousnessLevel::Transcendent => 1.0,
        }
    }
}

/// Attention mechanisms for consciousness-aware processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionMechanism {
    /// Current focus areas and their weights
    pub focus_weights: HashMap<String, f32>,
    /// Attention persistence over time
    pub attention_memory: Vec<(String, f32, chrono::DateTime<chrono::Utc>)>,
    /// Maximum attention capacity
    pub attention_capacity: usize,
    /// Attention decay rate
    pub decay_rate: f32,
}

impl AttentionMechanism {
    pub fn new(capacity: usize, decay_rate: f32) -> Self {
        Self {
            focus_weights: HashMap::new(),
            attention_memory: Vec::new(),
            attention_capacity: capacity,
            decay_rate,
        }
    }

    /// Update attention focus on a specific concept
    pub fn focus_on(&mut self, concept: &str, intensity: f32) {
        let now = chrono::Utc::now();

        // Add to current focus
        let current_weight = self.focus_weights.get(concept).unwrap_or(&0.0);
        let new_weight = (current_weight + intensity).min(1.0);
        self.focus_weights.insert(concept.to_string(), new_weight);

        // Add to attention memory
        self.attention_memory
            .push((concept.to_string(), intensity, now));

        // Maintain capacity limits
        if self.attention_memory.len() > self.attention_capacity {
            self.attention_memory.remove(0);
        }

        // Apply attention decay to older focuses
        self.apply_decay();
    }

    /// Apply attention decay over time
    fn apply_decay(&mut self) {
        let now = chrono::Utc::now();

        // Decay current focus weights
        for (_, weight) in self.focus_weights.iter_mut() {
            *weight *= (1.0 - self.decay_rate).max(0.0);
        }

        // Remove very weak attention
        self.focus_weights.retain(|_, &mut w| w > 0.01);

        // Decay attention memory based on time
        for (_, intensity, timestamp) in self.attention_memory.iter_mut() {
            let hours_passed = now.signed_duration_since(*timestamp).num_hours() as f32;
            let time_decay = (-hours_passed * 0.1).exp(); // Exponential decay
            *intensity *= time_decay;
        }

        // Remove very old or weak memories
        self.attention_memory
            .retain(|(_, intensity, _)| *intensity > 0.01);
    }

    /// Get current attention distribution
    pub fn get_attention_distribution(&self) -> Vec<(String, f32)> {
        let mut attention: Vec<_> = self
            .focus_weights
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        attention.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        attention
    }
}

/// Working memory for consciousness processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemory {
    /// Current concepts being processed
    pub active_concepts: HashMap<String, Array1<f32>>,
    /// Concept relationships
    pub concept_relationships: HashMap<(String, String), f32>,
    /// Working memory capacity (Miller's 7±2 rule)
    pub capacity: usize,
    /// Memory consolidation threshold
    pub consolidation_threshold: f32,
}

impl WorkingMemory {
    pub fn new(capacity: Option<usize>) -> Self {
        Self {
            active_concepts: HashMap::new(),
            concept_relationships: HashMap::new(),
            capacity: capacity.unwrap_or(7), // Miller's magical number
            consolidation_threshold: 0.7,
        }
    }

    /// Add a concept to working memory
    pub fn add_concept(&mut self, concept: String, embedding: Array1<f32>) -> Result<()> {
        // If at capacity, remove least active concept
        if self.active_concepts.len() >= self.capacity {
            self.remove_least_active()?;
        }

        self.active_concepts.insert(concept, embedding);
        Ok(())
    }

    /// Remove the least active concept
    fn remove_least_active(&mut self) -> Result<()> {
        // Simple strategy: remove first concept (could be improved with usage tracking)
        if let Some(concept) = self.active_concepts.keys().next().cloned() {
            self.active_concepts.remove(&concept);
        }
        Ok(())
    }

    /// Update concept relationships
    pub fn update_relationship(&mut self, concept1: &str, concept2: &str, strength: f32) {
        let key = if concept1 < concept2 {
            (concept1.to_string(), concept2.to_string())
        } else {
            (concept2.to_string(), concept1.to_string())
        };

        self.concept_relationships.insert(key, strength);
    }

    /// Get related concepts
    pub fn get_related_concepts(&self, concept: &str) -> Vec<(String, f32)> {
        self.concept_relationships
            .iter()
            .filter_map(|((c1, c2), &strength)| {
                if c1 == concept {
                    Some((c2.clone(), strength))
                } else if c2 == concept {
                    Some((c1.clone(), strength))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Meta-cognitive layer for self-awareness and reflection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaCognition {
    /// Self-assessment of current understanding
    pub understanding_confidence: f32,
    /// Awareness of knowledge gaps
    pub knowledge_gaps: Vec<String>,
    /// Strategy effectiveness tracking
    pub strategy_effectiveness: HashMap<String, f32>,
    /// Self-reflection notes
    pub reflection_history: Vec<(chrono::DateTime<chrono::Utc>, String, f32)>,
}

impl MetaCognition {
    pub fn new() -> Self {
        Self {
            understanding_confidence: 0.5,
            knowledge_gaps: Vec::new(),
            strategy_effectiveness: HashMap::new(),
            reflection_history: Vec::new(),
        }
    }

    /// Update confidence based on recent performance
    pub fn update_confidence(&mut self, performance_score: f32) {
        let alpha = 0.1; // Learning rate for confidence updates
        self.understanding_confidence =
            (1.0 - alpha) * self.understanding_confidence + alpha * performance_score;
        self.understanding_confidence = self.understanding_confidence.clamp(0.0, 1.0);
    }

    /// Add a reflection note
    pub fn reflect(&mut self, insight: String, confidence: f32) {
        let now = chrono::Utc::now();
        self.reflection_history.push((now, insight, confidence));

        // Keep only recent reflections
        if self.reflection_history.len() > 100 {
            self.reflection_history.remove(0);
        }
    }

    /// Identify knowledge gaps
    pub fn identify_knowledge_gap(&mut self, domain: String) {
        if !self.knowledge_gaps.contains(&domain) {
            self.knowledge_gaps.push(domain);
        }
    }

    /// Update strategy effectiveness
    pub fn update_strategy_effectiveness(&mut self, strategy: &str, effectiveness: f32) {
        let current = self.strategy_effectiveness.get(strategy).unwrap_or(&0.5);
        let alpha = 0.2;
        let new_effectiveness = (1.0 - alpha) * current + alpha * effectiveness;
        self.strategy_effectiveness
            .insert(strategy.to_string(), new_effectiveness);
    }
}

impl Default for MetaCognition {
    fn default() -> Self {
        Self::new()
    }
}

/// Main consciousness-aware embedding model
#[derive(Debug)]
pub struct ConsciousnessAwareEmbedding {
    /// Unique model identifier
    pub model_id: Uuid,
    /// Model configuration
    pub config: ModelConfig,
    /// Current consciousness level
    pub consciousness_level: ConsciousnessLevel,
    /// Attention mechanism
    pub attention: AttentionMechanism,
    /// Working memory
    pub working_memory: WorkingMemory,
    /// Meta-cognitive layer
    pub meta_cognition: MetaCognition,
    /// Entity embeddings with consciousness enhancement
    pub embeddings: HashMap<String, Array1<f32>>,
    /// Consciousness state vector
    pub consciousness_state: Array1<f32>,
    /// Learning history for adaptation
    pub learning_history: Vec<(String, Array1<f32>, f32)>, // (concept, embedding, quality)
}

impl ConsciousnessAwareEmbedding {
    pub fn new(config: ModelConfig) -> Self {
        let dimensions = config.dimensions;

        Self {
            model_id: Uuid::new_v4(),
            config,
            consciousness_level: ConsciousnessLevel::SelfAware,
            attention: AttentionMechanism::new(20, 0.05),
            working_memory: WorkingMemory::new(Some(7)),
            meta_cognition: MetaCognition::new(),
            embeddings: HashMap::new(),
            consciousness_state: Array1::zeros(dimensions),
            learning_history: Vec::new(),
        }
    }

    /// Generate consciousness-aware embedding for an entity
    pub fn generate_conscious_embedding(
        &mut self,
        entity: &str,
        context: &[String],
    ) -> Result<Array1<f32>> {
        // Update attention based on context
        for ctx in context {
            self.attention.focus_on(ctx, 0.3);
        }
        self.attention.focus_on(entity, 0.8);

        // Get base embedding or create new one
        let base_embedding = self
            .embeddings
            .get(entity)
            .cloned()
            .unwrap_or_else(|| self.create_base_embedding(entity));

        // Apply consciousness-aware modifications
        let conscious_embedding =
            self.apply_consciousness_enhancement(&base_embedding, entity, context)?;

        // Update working memory
        self.working_memory
            .add_concept(entity.to_string(), conscious_embedding.clone())?;

        // Store enhanced embedding
        self.embeddings
            .insert(entity.to_string(), conscious_embedding.clone());

        // Meta-cognitive reflection
        self.reflect_on_embedding(entity, &conscious_embedding);

        Ok(conscious_embedding)
    }

    /// Create a base embedding for an entity
    fn create_base_embedding(&self, entity: &str) -> Array1<f32> {
        let dimensions = self.config.dimensions;

        // Simple hash-based initialization (could be more sophisticated)
        let hash = entity.chars().map(|c| c as u32).sum::<u32>();
        let _seed = hash as u64;

        {
            #[allow(unused_imports)]
            use scirs2_core::random::{Random, Rng};
            // Note: Seeding not supported in scirs2_core - would be deterministic if needed
            Array1::from_shape_fn(dimensions, |_| {
                let mut random = Random::default();
                random.gen_range(-0.1..0.1)
            })
        }
    }

    /// Apply consciousness enhancement to base embedding
    fn apply_consciousness_enhancement(
        &mut self,
        base_embedding: &Array1<f32>,
        entity: &str,
        context: &[String],
    ) -> Result<Array1<f32>> {
        let mut enhanced = base_embedding.clone();

        // Apply attention-based modulation
        let attention_factor = self.attention.focus_weights.get(entity).unwrap_or(&0.0)
            * self.consciousness_level.awareness_factor();

        // Consciousness state influence
        let consciousness_influence = &self.consciousness_state * attention_factor;
        enhanced = enhanced + consciousness_influence;

        // Context integration
        for ctx in context {
            if let Some(ctx_embedding) = self.embeddings.get(ctx) {
                let context_weight = self.attention.focus_weights.get(ctx).unwrap_or(&0.0) * 0.2;
                enhanced = enhanced + ctx_embedding * context_weight;
            }
        }

        // Working memory integration
        let related_concepts = self.working_memory.get_related_concepts(entity);
        for (related_concept, relationship_strength) in related_concepts {
            if let Some(related_embedding) =
                self.working_memory.active_concepts.get(&related_concept)
            {
                enhanced = enhanced + related_embedding * (relationship_strength * 0.1);
            }
        }

        // Meta-cognitive adjustment based on confidence
        let confidence_factor = self.meta_cognition.understanding_confidence;
        enhanced = enhanced * confidence_factor + base_embedding * (1.0 - confidence_factor);

        // Normalize to prevent explosion
        let norm = enhanced.mapv(|x| x * x).sum().sqrt();
        if norm > 0.0 {
            enhanced /= norm;
        }

        Ok(enhanced)
    }

    /// Meta-cognitive reflection on generated embedding
    fn reflect_on_embedding(&mut self, entity: &str, embedding: &Array1<f32>) {
        // Analyze embedding quality
        let norm = embedding.mapv(|x| x * x).sum().sqrt();
        let quality_score = if norm > 0.0 && norm < 2.0 {
            (1.0 - (norm - 1.0).abs()).max(0.0)
        } else {
            0.0
        };

        // Update meta-cognition
        self.meta_cognition.update_confidence(quality_score);

        // Reflect on the process
        let reflection = format!(
            "Generated embedding for '{entity}' with quality {quality_score:.3} and norm {norm:.3}"
        );
        self.meta_cognition.reflect(reflection, quality_score);

        // Add to learning history
        self.learning_history
            .push((entity.to_string(), embedding.clone(), quality_score));

        // Keep learning history manageable
        if self.learning_history.len() > 1000 {
            self.learning_history.remove(0);
        }
    }

    /// Elevate consciousness level based on experience
    pub fn evolve_consciousness(&mut self) -> Result<()> {
        let total_experience = self.learning_history.len();
        let average_quality = if !self.learning_history.is_empty() {
            self.learning_history.iter().map(|(_, _, q)| q).sum::<f32>() / total_experience as f32
        } else {
            0.0
        };

        // Determine if consciousness should evolve
        let evolution_threshold = self.consciousness_level.complexity_threshold();

        if average_quality > evolution_threshold && total_experience > 100 {
            self.consciousness_level = match self.consciousness_level {
                ConsciousnessLevel::Reactive => ConsciousnessLevel::Associative,
                ConsciousnessLevel::Associative => ConsciousnessLevel::SelfAware,
                ConsciousnessLevel::SelfAware => ConsciousnessLevel::Reflective,
                ConsciousnessLevel::Reflective => ConsciousnessLevel::MetaCognitive,
                ConsciousnessLevel::MetaCognitive => ConsciousnessLevel::Transcendent,
                ConsciousnessLevel::Transcendent => ConsciousnessLevel::Transcendent, // Stays at highest
            };

            // Update consciousness state vector
            self.update_consciousness_state()?;

            // Meta-cognitive reflection on evolution
            self.meta_cognition.reflect(
                format!("Consciousness evolved to {:?}", self.consciousness_level),
                0.9,
            );
        }

        Ok(())
    }

    /// Update consciousness state vector
    fn update_consciousness_state(&mut self) -> Result<()> {
        let dimensions = self.consciousness_state.len();
        let awareness_factor = self.consciousness_level.awareness_factor();

        // Update consciousness state based on recent experiences
        if !self.learning_history.is_empty() {
            let recent_embeddings = self
                .learning_history
                .iter()
                .rev()
                .take(50) // Last 50 experiences
                .map(|(_, emb, quality)| emb * *quality)
                .fold(Array1::<f32>::zeros(dimensions), |acc, emb| acc + emb);

            let count = self.learning_history.len().min(50) as f32;
            let average_recent = recent_embeddings / count;

            // Integrate with existing consciousness state
            self.consciousness_state =
                &self.consciousness_state * 0.8f32 + &average_recent * 0.2f32;

            // Apply consciousness-level scaling
            self.consciousness_state = &self.consciousness_state * awareness_factor;
        }

        Ok(())
    }

    /// Get consciousness insights for debugging/monitoring
    pub fn get_consciousness_insights(&self) -> ConsciousnessInsights {
        let attention_distribution = self.attention.get_attention_distribution();
        let active_concepts: Vec<String> = self
            .working_memory
            .active_concepts
            .keys()
            .cloned()
            .collect();
        let recent_reflections: Vec<String> = self
            .meta_cognition
            .reflection_history
            .iter()
            .rev()
            .take(5)
            .map(|(_, reflection, _)| reflection.clone())
            .collect();

        ConsciousnessInsights {
            consciousness_level: self.consciousness_level,
            understanding_confidence: self.meta_cognition.understanding_confidence,
            attention_distribution,
            active_concepts,
            recent_reflections,
            total_experiences: self.learning_history.len(),
            knowledge_gaps: self.meta_cognition.knowledge_gaps.clone(),
        }
    }
}

/// Consciousness insights for monitoring and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessInsights {
    pub consciousness_level: ConsciousnessLevel,
    pub understanding_confidence: f32,
    pub attention_distribution: Vec<(String, f32)>,
    pub active_concepts: Vec<String>,
    pub recent_reflections: Vec<String>,
    pub total_experiences: usize,
    pub knowledge_gaps: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_levels() {
        assert_eq!(ConsciousnessLevel::Reactive.awareness_factor(), 0.1);
        assert_eq!(ConsciousnessLevel::Transcendent.awareness_factor(), 1.0);
    }

    #[test]
    fn test_attention_mechanism() {
        let mut attention = AttentionMechanism::new(10, 0.1);

        attention.focus_on("concept1", 0.8);
        attention.focus_on("concept2", 0.6);

        let distribution = attention.get_attention_distribution();
        assert_eq!(distribution.len(), 2);
        assert_eq!(distribution[0].0, "concept1");
        assert!(distribution[0].1 > distribution[1].1);
    }

    #[test]
    fn test_working_memory() {
        let mut memory = WorkingMemory::new(Some(3));

        let emb1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let emb2 = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        assert!(memory.add_concept("concept1".to_string(), emb1).is_ok());
        assert!(memory.add_concept("concept2".to_string(), emb2).is_ok());

        memory.update_relationship("concept1", "concept2", 0.8);
        let related = memory.get_related_concepts("concept1");
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].0, "concept2");
        assert_eq!(related[0].1, 0.8);
    }

    #[test]
    fn test_consciousness_aware_embedding() {
        let config = ModelConfig::default().with_dimensions(64);
        let mut model = ConsciousnessAwareEmbedding::new(config);

        let context = vec!["related_entity".to_string()];
        let embedding = model.generate_conscious_embedding("test_entity", &context);

        assert!(embedding.is_ok());
        let emb = embedding.unwrap();
        assert_eq!(emb.len(), 64);

        // Check that attention was updated
        assert!(model.attention.focus_weights.contains_key("test_entity"));

        // Check working memory
        assert!(model
            .working_memory
            .active_concepts
            .contains_key("test_entity"));
    }

    #[test]
    fn test_consciousness_evolution() {
        let config = ModelConfig::default().with_dimensions(32);
        let mut model = ConsciousnessAwareEmbedding::new(config);

        // Add some quality experiences
        for i in 0..150 {
            let entity = format!("entity_{i}");
            let embedding = Array1::from_vec(vec![0.1; 32]);
            model.learning_history.push((entity, embedding, 0.9)); // High quality
        }

        let initial_level = model.consciousness_level;
        assert!(model.evolve_consciousness().is_ok());

        // Should have evolved due to high-quality experiences
        assert_ne!(model.consciousness_level, initial_level);
    }

    #[tokio::test]
    async fn test_consciousness_insights() {
        let config = ModelConfig::default().with_dimensions(16);
        let mut model = ConsciousnessAwareEmbedding::new(config);

        // Generate some embeddings to create insights
        let context = vec!["context1".to_string()];
        let _ = model.generate_conscious_embedding("entity1", &context);
        let _ = model.generate_conscious_embedding("entity2", &context);

        let insights = model.get_consciousness_insights();

        assert_eq!(insights.consciousness_level, ConsciousnessLevel::SelfAware);
        assert!(insights.total_experiences > 0);
        assert!(!insights.active_concepts.is_empty());
        assert!(!insights.attention_distribution.is_empty());
    }
}
