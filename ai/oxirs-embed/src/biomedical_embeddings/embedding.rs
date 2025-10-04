//! Module for biomedical embeddings

use super::*;
use crate::{EmbeddingModel, ModelConfig, ModelStats, TrainingStats, Triple, Vector};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::Utc;
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::random::{Random, Rng};
use std::collections::HashMap;
use uuid::Uuid;

impl BiomedicalEmbedding {
    /// Create new biomedical embedding model
    pub fn new(config: BiomedicalEmbeddingConfig) -> Self {
        let model_id = Uuid::new_v4();
        let now = Utc::now();

        Self {
            model_id,
            gene_embeddings: HashMap::new(),
            protein_embeddings: HashMap::new(),
            disease_embeddings: HashMap::new(),
            drug_embeddings: HashMap::new(),
            compound_embeddings: HashMap::new(),
            pathway_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            entity_types: HashMap::new(),
            relation_types: HashMap::new(),
            triples: Vec::new(),
            features: BiomedicalFeatures::default(),
            training_stats: TrainingStats::default(),
            model_stats: ModelStats {
                num_entities: 0,
                num_relations: 0,
                num_triples: 0,
                dimensions: config.base_config.dimensions,
                is_trained: false,
                model_type: "BiomedicalEmbedding".to_string(),
                creation_time: now,
                last_training_time: None,
            },
            is_trained: false,
            config,
        }
    }

    /// Get the model type identifier
    pub fn model_type(&self) -> &str {
        "BiomedicalEmbedding"
    }

    /// Check if the model has been trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Add gene-disease association
    pub fn add_gene_disease_association(&mut self, gene: &str, disease: &str, score: f32) {
        self.features
            .gene_disease_associations
            .insert((gene.to_string(), disease.to_string()), score);

        // Also add reverse mapping
        self.features
            .gene_disease_associations
            .insert((disease.to_string(), gene.to_string()), score);
    }

    /// Add drug-target interaction
    pub fn add_drug_target_interaction(&mut self, drug: &str, target: &str, affinity: f32) {
        self.features
            .drug_target_affinities
            .insert((drug.to_string(), target.to_string()), affinity);
    }

    /// Add pathway membership
    pub fn add_pathway_membership(&mut self, entity: &str, pathway: &str, score: f32) {
        self.features
            .pathway_memberships
            .insert((entity.to_string(), pathway.to_string()), score);
    }

    /// Add protein-protein interaction
    pub fn add_protein_interaction(&mut self, protein1: &str, protein2: &str, score: f32) {
        self.features
            .protein_interactions
            .insert((protein1.to_string(), protein2.to_string()), score);

        // Symmetric relationship
        self.features
            .protein_interactions
            .insert((protein2.to_string(), protein1.to_string()), score);
    }

    /// Get entity embedding with biomedical type awareness
    pub fn get_typed_entity_embedding(&self, entity: &str) -> Result<Vector> {
        if let Some(entity_type) = self.entity_types.get(entity) {
            let embedding = match entity_type {
                BiomedicalEntityType::Gene => self.gene_embeddings.get(entity),
                BiomedicalEntityType::Protein => self.protein_embeddings.get(entity),
                BiomedicalEntityType::Disease => self.disease_embeddings.get(entity),
                BiomedicalEntityType::Drug => self.drug_embeddings.get(entity),
                BiomedicalEntityType::Compound => self.compound_embeddings.get(entity),
                BiomedicalEntityType::Pathway => self.pathway_embeddings.get(entity),
                _ => None,
            };

            if let Some(emb) = embedding {
                Ok(Vector::from_array1(emb))
            } else {
                Err(anyhow!(
                    "No embedding found for {} of type {:?}",
                    entity,
                    entity_type
                ))
            }
        } else {
            Err(anyhow!("Unknown entity type for {}", entity))
        }
    }

    /// Predict gene-disease associations
    pub fn predict_gene_disease_associations(
        &self,
        gene: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        if !self.is_trained {
            return Err(anyhow!("Model not trained"));
        }

        let gene_embedding = self
            .gene_embeddings
            .get(gene)
            .ok_or_else(|| anyhow!("Gene {} not found", gene))?;

        let mut scores = Vec::new();

        for (disease, disease_embedding) in &self.disease_embeddings {
            // Base similarity
            let similarity = gene_embedding.dot(disease_embedding) as f64;

            // Enhance with existing association data
            let enhanced_score = if let Some(&assoc_score) = self
                .features
                .gene_disease_associations
                .get(&(gene.to_string(), disease.clone()))
            {
                similarity * (1.0 + assoc_score as f64)
            } else {
                similarity
            };

            scores.push((disease.clone(), enhanced_score));
        }

        // Sort by score and return top k
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    /// Predict drug targets
    pub fn predict_drug_targets(&self, drug: &str, k: usize) -> Result<Vec<(String, f64)>> {
        if !self.is_trained {
            return Err(anyhow!("Model not trained"));
        }

        let drug_embedding = self
            .drug_embeddings
            .get(drug)
            .ok_or_else(|| anyhow!("Drug {} not found", drug))?;

        let mut scores = Vec::new();

        for (protein, protein_embedding) in &self.protein_embeddings {
            // Base similarity
            let similarity = drug_embedding.dot(protein_embedding) as f64;

            // Enhance with binding affinity data
            let enhanced_score = if let Some(&affinity) = self
                .features
                .drug_target_affinities
                .get(&(drug.to_string(), protein.clone()))
            {
                similarity * (1.0 + affinity as f64)
            } else {
                similarity
            };

            scores.push((protein.clone(), enhanced_score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    /// Find pathway-related entities
    pub fn find_pathway_entities(&self, pathway: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let pathway_embedding = self
            .pathway_embeddings
            .get(pathway)
            .ok_or_else(|| anyhow!("Pathway {} not found", pathway))?;

        let mut scores = Vec::new();

        // Check genes
        for (gene, gene_embedding) in &self.gene_embeddings {
            let similarity = pathway_embedding.dot(gene_embedding) as f64;
            scores.push((gene.clone(), similarity));
        }

        // Check proteins
        for (protein, protein_embedding) in &self.protein_embeddings {
            let similarity = pathway_embedding.dot(protein_embedding) as f64;
            scores.push((protein.clone(), similarity));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    /// Extract entity types from triples
    fn extract_entity_types(&mut self) {
        for triple in &self.triples {
            // Extract entity types from IRIs
            if let Some(subject_type) = BiomedicalEntityType::from_iri(&triple.subject.iri) {
                self.entity_types
                    .insert(triple.subject.iri.clone(), subject_type);
            }

            if let Some(object_type) = BiomedicalEntityType::from_iri(&triple.object.iri) {
                self.entity_types
                    .insert(triple.object.iri.clone(), object_type);
            }

            // Extract relation types
            if let Some(relation_type) = BiomedicalRelationType::from_iri(&triple.predicate.iri) {
                self.relation_types
                    .insert(triple.predicate.iri.clone(), relation_type);
            }
        }
    }

    /// Initialize embeddings with biomedical-specific features
    fn initialize_embeddings(&mut self) -> Result<()> {
        let dimensions = self.config.base_config.dimensions;

        // Initialize embeddings for each entity type
        for (entity, entity_type) in &self.entity_types {
            let embedding = Array1::from_vec(
                (0..dimensions)
                    .map(|_| {
                        let mut random = Random::default();
                        (random.random::<f32>() - 0.5) * 0.1
                    })
                    .collect(),
            );

            match entity_type {
                BiomedicalEntityType::Gene => {
                    self.gene_embeddings.insert(entity.clone(), embedding);
                }
                BiomedicalEntityType::Protein => {
                    self.protein_embeddings.insert(entity.clone(), embedding);
                }
                BiomedicalEntityType::Disease => {
                    self.disease_embeddings.insert(entity.clone(), embedding);
                }
                BiomedicalEntityType::Drug => {
                    self.drug_embeddings.insert(entity.clone(), embedding);
                }
                BiomedicalEntityType::Compound => {
                    self.compound_embeddings.insert(entity.clone(), embedding);
                }
                BiomedicalEntityType::Pathway => {
                    self.pathway_embeddings.insert(entity.clone(), embedding);
                }
                _ => {
                    // For other types, store in a general embedding map
                    // This would be extended in a full implementation
                }
            }
        }

        // Initialize relation embeddings
        for relation in self.relation_types.keys() {
            let embedding = Array1::from_vec(
                (0..dimensions)
                    .map(|_| {
                        let mut random = Random::default();
                        (random.random::<f32>() - 0.5) * 0.1
                    })
                    .collect(),
            );
            self.relation_embeddings.insert(relation.clone(), embedding);
        }

        Ok(())
    }

    /// Compute biomedical-specific loss incorporating domain knowledge
    fn compute_biomedical_loss(&self) -> f32 {
        let mut total_loss = 0.0;
        let mut count = 0;

        // Gene-disease association loss
        for ((gene, disease), &score) in &self.features.gene_disease_associations {
            if let (Some(gene_emb), Some(disease_emb)) = (
                self.gene_embeddings.get(gene),
                self.disease_embeddings.get(disease),
            ) {
                let predicted_score = gene_emb.dot(disease_emb);
                let loss = (predicted_score - score).powi(2);
                total_loss += loss * self.config.gene_disease_weight;
                count += 1;
            }
        }

        // Drug-target interaction loss
        for ((drug, target), &affinity) in &self.features.drug_target_affinities {
            if let (Some(drug_emb), Some(target_emb)) = (
                self.drug_embeddings.get(drug),
                self.protein_embeddings.get(target),
            ) {
                let predicted_affinity = drug_emb.dot(target_emb);
                let loss = (predicted_affinity - affinity).powi(2);
                total_loss += loss * self.config.drug_target_weight;
                count += 1;
            }
        }

        // Pathway membership loss
        for ((entity, pathway), &score) in &self.features.pathway_memberships {
            if let Some(pathway_emb) = self.pathway_embeddings.get(pathway) {
                let entity_emb = self.get_entity_embedding_any_type(entity);
                if let Some(entity_emb) = entity_emb {
                    let predicted_score = entity_emb.dot(pathway_emb);
                    let loss = (predicted_score - score).powi(2);
                    total_loss += loss * self.config.pathway_weight;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }

    /// Helper to get entity embedding from any type map
    fn get_entity_embedding_any_type(&self, entity: &str) -> Option<&Array1<f32>> {
        self.gene_embeddings
            .get(entity)
            .or_else(|| self.protein_embeddings.get(entity))
            .or_else(|| self.disease_embeddings.get(entity))
            .or_else(|| self.drug_embeddings.get(entity))
            .or_else(|| self.compound_embeddings.get(entity))
            .or_else(|| self.pathway_embeddings.get(entity))
    }
}

#[async_trait]
impl EmbeddingModel for BiomedicalEmbedding {
    fn config(&self) -> &ModelConfig {
        &self.config.base_config
    }

    fn model_id(&self) -> &Uuid {
        &self.model_id
    }

    fn model_type(&self) -> &'static str {
        "BiomedicalEmbedding"
    }

    fn add_triple(&mut self, triple: Triple) -> Result<()> {
        self.triples.push(triple);
        Ok(())
    }

    async fn train(&mut self, epochs: Option<usize>) -> Result<TrainingStats> {
        let epochs = epochs.unwrap_or(1000);
        let start_time = std::time::Instant::now();

        // Extract entity and relation types
        self.extract_entity_types();

        // Initialize embeddings
        self.initialize_embeddings()?;

        // Training loop
        let mut loss_history = Vec::new();

        for epoch in 0..epochs {
            let epoch_loss = self.compute_biomedical_loss();
            loss_history.push(epoch_loss as f64);

            // Simple convergence check
            if epoch > 10 && epoch_loss < 0.001 {
                break;
            }

            if epoch % 100 == 0 {
                println!("Epoch {epoch}: Loss = {epoch_loss:.6}");
            }
        }

        let training_time = start_time.elapsed().as_secs_f64();

        self.training_stats = TrainingStats {
            epochs_completed: epochs,
            final_loss: loss_history.last().copied().unwrap_or(0.0),
            training_time_seconds: training_time,
            convergence_achieved: loss_history.last().is_some_and(|&loss| loss < 0.001),
            loss_history,
        };

        self.is_trained = true;
        self.model_stats.is_trained = true;
        self.model_stats.last_training_time = Some(Utc::now());

        // Update entity counts
        self.model_stats.num_entities = self.entity_types.len();
        self.model_stats.num_relations = self.relation_types.len();
        self.model_stats.num_triples = self.triples.len();

        Ok(self.training_stats.clone())
    }

    fn get_entity_embedding(&self, entity: &str) -> Result<Vector> {
        self.get_typed_entity_embedding(entity)
    }

    fn getrelation_embedding(&self, relation: &str) -> Result<Vector> {
        if let Some(embedding) = self.relation_embeddings.get(relation) {
            Ok(Vector::from_array1(embedding))
        } else {
            Err(anyhow!("Relation {} not found", relation))
        }
    }

    fn score_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<f64> {
        let subject_emb = self.get_entity_embedding(subject)?;
        let relation_emb = self.getrelation_embedding(predicate)?;
        let object_emb = self.get_entity_embedding(object)?;

        // TransE-style scoring with biomedical enhancements
        let mut score = 0.0;
        for i in 0..subject_emb.dimensions {
            let diff = subject_emb.values[i] + relation_emb.values[i] - object_emb.values[i];
            score += diff * diff;
        }

        // Convert to similarity score (higher is better)
        Ok(1.0 / (1.0 + score as f64))
    }

    fn predict_objects(
        &self,
        subject: &str,
        predicate: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        // Use specialized prediction methods based on relation type
        if let Some(relation_type) = self.relation_types.get(predicate) {
            match relation_type {
                BiomedicalRelationType::CausesDisease
                | BiomedicalRelationType::AssociatedWithDisease => {
                    return self.predict_gene_disease_associations(subject, k);
                }
                BiomedicalRelationType::TargetsProtein | BiomedicalRelationType::BindsToProtein => {
                    return self.predict_drug_targets(subject, k);
                }
                _ => {
                    // Fall back to generic prediction
                }
            }
        }

        // Generic prediction
        let _subject_emb = self.get_entity_embedding(subject)?;
        let _relation_emb = self.getrelation_embedding(predicate)?;

        let mut scores = Vec::new();
        for entity in self.entity_types.keys() {
            if entity != subject {
                if let Ok(score) = self.score_triple(subject, predicate, entity) {
                    scores.push((entity.clone(), score));
                }
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    fn predict_subjects(
        &self,
        predicate: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let _object_emb = self.get_entity_embedding(object)?;
        let _relation_emb = self.getrelation_embedding(predicate)?;

        let mut scores = Vec::new();
        for entity in self.entity_types.keys() {
            if entity != object {
                if let Ok(score) = self.score_triple(entity, predicate, object) {
                    scores.push((entity.clone(), score));
                }
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    fn predict_relations(
        &self,
        subject: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let _subject_emb = self.get_entity_embedding(subject)?;
        let _object_emb = self.get_entity_embedding(object)?;

        let mut scores = Vec::new();
        for relation in self.relation_types.keys() {
            if let Ok(score) = self.score_triple(subject, relation, object) {
                scores.push((relation.clone(), score));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    fn get_entities(&self) -> Vec<String> {
        self.entity_types.keys().cloned().collect()
    }

    fn get_relations(&self) -> Vec<String> {
        self.relation_types.keys().cloned().collect()
    }

    fn get_stats(&self) -> ModelStats {
        self.model_stats.clone()
    }

    fn save(&self, _path: &str) -> Result<()> {
        // Implementation would serialize the model
        Ok(())
    }

    fn load(&mut self, _path: &str) -> Result<()> {
        // Implementation would deserialize the model
        Ok(())
    }

    fn clear(&mut self) {
        self.gene_embeddings.clear();
        self.protein_embeddings.clear();
        self.disease_embeddings.clear();
        self.drug_embeddings.clear();
        self.compound_embeddings.clear();
        self.pathway_embeddings.clear();
        self.relation_embeddings.clear();
        self.entity_types.clear();
        self.relation_types.clear();
        self.triples.clear();
        self.features = BiomedicalFeatures::default();
        self.is_trained = false;
    }

    fn is_trained(&self) -> bool {
        self.is_trained
    }

    async fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            match self.get_entity_embedding(text) {
                Ok(embedding) => {
                    embeddings.push(embedding.values);
                }
                _ => {
                    // Return zero embedding for unknown entities
                    embeddings.push(vec![0.0; self.config.base_config.dimensions]);
                }
            }
        }

        Ok(embeddings)
    }
}
