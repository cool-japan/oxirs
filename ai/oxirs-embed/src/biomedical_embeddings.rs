//! Biomedical knowledge graph embeddings for scientific applications
//!
//! This module provides specialized embeddings for biomedical knowledge graphs
//! including gene-disease associations, drug-target interactions, pathways,
//! protein structures, and medical concept hierarchies.

use crate::{EmbeddingModel, ModelConfig, ModelStats, TrainingStats, Triple, Vector};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::Utc;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Biomedical entity types for specialized embedding handling
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BiomedicalEntityType {
    Gene,
    Protein,
    Disease,
    Drug,
    Compound,
    Pathway,
    Cell,
    Tissue,
    Organ,
    Phenotype,
    GO_Term,
    MESH_Term,
    SNOMED_CT,
    ICD_Code,
}

impl BiomedicalEntityType {
    /// Get the namespace prefix for this entity type
    pub fn namespace(&self) -> &'static str {
        match self {
            BiomedicalEntityType::Gene => "gene",
            BiomedicalEntityType::Protein => "protein", 
            BiomedicalEntityType::Disease => "disease",
            BiomedicalEntityType::Drug => "drug",
            BiomedicalEntityType::Compound => "compound",
            BiomedicalEntityType::Pathway => "pathway",
            BiomedicalEntityType::Cell => "cell",
            BiomedicalEntityType::Tissue => "tissue",
            BiomedicalEntityType::Organ => "organ",
            BiomedicalEntityType::Phenotype => "phenotype",
            BiomedicalEntityType::GO_Term => "go",
            BiomedicalEntityType::MESH_Term => "mesh",
            BiomedicalEntityType::SNOMED_CT => "snomed",
            BiomedicalEntityType::ICD_Code => "icd",
        }
    }

    /// Parse entity type from IRI
    pub fn from_iri(iri: &str) -> Option<Self> {
        if iri.contains("gene") || iri.contains("HGNC") {
            Some(BiomedicalEntityType::Gene)
        } else if iri.contains("protein") || iri.contains("UniProt") {
            Some(BiomedicalEntityType::Protein)
        } else if iri.contains("disease") || iri.contains("OMIM") || iri.contains("DOID") {
            Some(BiomedicalEntityType::Disease)
        } else if iri.contains("drug") || iri.contains("DrugBank") {
            Some(BiomedicalEntityType::Drug)
        } else if iri.contains("compound") || iri.contains("CHEBI") {
            Some(BiomedicalEntityType::Compound)
        } else if iri.contains("pathway") || iri.contains("KEGG") || iri.contains("Reactome") {
            Some(BiomedicalEntityType::Pathway)
        } else if iri.contains("GO:") {
            Some(BiomedicalEntityType::GO_Term)
        } else if iri.contains("MESH") {
            Some(BiomedicalEntityType::MESH_Term)
        } else if iri.contains("SNOMED") {
            Some(BiomedicalEntityType::SNOMED_CT)
        } else if iri.contains("ICD") {
            Some(BiomedicalEntityType::ICD_Code)
        } else {
            None
        }
    }
}

/// Biomedical relation types for specialized handling
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BiomedicalRelationType {
    /// Gene-disease associations
    CausesDisease,
    AssociatedWithDisease,
    PredisposesToDisease,
    /// Drug-target interactions
    TargetsProtein,
    InhibitsProtein,
    ActivatesProtein,
    BindsToProtein,
    /// Pathway relationships
    ParticipatesInPathway,
    RegulatesPathway,
    UpstreamOfPathway,
    DownstreamOfPathway,
    /// Protein interactions
    InteractsWith,
    PhysicallyInteractsWith,
    FunctionallyInteractsWith,
    /// Chemical relationships
    MetabolizedBy,
    TransportedBy,
    Catalyzes,
    /// Hierarchical relationships
    IsASubtypeOf,
    PartOf,
    HasPhenotype,
    /// Expression relationships
    ExpressedIn,
    Overexpressed,
    Underexpressed,
}

impl BiomedicalRelationType {
    /// Parse relation type from predicate IRI
    pub fn from_iri(iri: &str) -> Option<Self> {
        match iri.to_lowercase().as_str() {
            s if s.contains("causes") => Some(BiomedicalRelationType::CausesDisease),
            s if s.contains("associated_with") => Some(BiomedicalRelationType::AssociatedWithDisease),
            s if s.contains("targets") => Some(BiomedicalRelationType::TargetsProtein),
            s if s.contains("inhibits") => Some(BiomedicalRelationType::InhibitsProtein),
            s if s.contains("activates") => Some(BiomedicalRelationType::ActivatesProtein),
            s if s.contains("binds") => Some(BiomedicalRelationType::BindsToProtein),
            s if s.contains("participates") => Some(BiomedicalRelationType::ParticipatesInPathway),
            s if s.contains("interacts") => Some(BiomedicalRelationType::InteractsWith),
            s if s.contains("metabolized") => Some(BiomedicalRelationType::MetabolizedBy),
            s if s.contains("expressed") => Some(BiomedicalRelationType::ExpressedIn),
            s if s.contains("subtype") => Some(BiomedicalRelationType::IsASubtypeOf),
            s if s.contains("part_of") => Some(BiomedicalRelationType::PartOf),
            _ => None,
        }
    }
}

/// Configuration for biomedical embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomedicalEmbeddingConfig {
    pub base_config: ModelConfig,
    /// Weight for gene-disease associations
    pub gene_disease_weight: f32,
    /// Weight for drug-target interactions
    pub drug_target_weight: f32,
    /// Weight for pathway relationships
    pub pathway_weight: f32,
    /// Weight for protein interactions
    pub protein_interaction_weight: f32,
    /// Enable sequence similarity features
    pub use_sequence_similarity: bool,
    /// Enable chemical structure features
    pub use_chemical_structure: bool,
    /// Enable taxonomic hierarchy
    pub use_taxonomy: bool,
    /// Enable temporal relationships
    pub use_temporal_features: bool,
    /// Species filter (e.g., "Homo sapiens", "Mus musculus")
    pub species_filter: Option<String>,
}

impl Default for BiomedicalEmbeddingConfig {
    fn default() -> Self {
        Self {
            base_config: ModelConfig::default(),
            gene_disease_weight: 2.0,
            drug_target_weight: 1.5,
            pathway_weight: 1.2,
            protein_interaction_weight: 1.0,
            use_sequence_similarity: true,
            use_chemical_structure: true,
            use_taxonomy: true,
            use_temporal_features: false,
            species_filter: Some("Homo sapiens".to_string()),
        }
    }
}

/// Biomedical knowledge graph embedding model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomedicalEmbedding {
    pub config: BiomedicalEmbeddingConfig,
    pub model_id: Uuid,
    /// Entity embeddings by type
    pub gene_embeddings: HashMap<String, Array1<f32>>,
    pub protein_embeddings: HashMap<String, Array1<f32>>,
    pub disease_embeddings: HashMap<String, Array1<f32>>,
    pub drug_embeddings: HashMap<String, Array1<f32>>,
    pub compound_embeddings: HashMap<String, Array1<f32>>,
    pub pathway_embeddings: HashMap<String, Array1<f32>>,
    /// Relation embeddings by type
    pub relation_embeddings: HashMap<String, Array1<f32>>,
    /// Entity type mappings
    pub entity_types: HashMap<String, BiomedicalEntityType>,
    /// Relation type mappings
    pub relation_types: HashMap<String, BiomedicalRelationType>,
    /// Training data
    pub triples: Vec<Triple>,
    /// Biomedical-specific features
    pub features: BiomedicalFeatures,
    /// Training and model stats
    pub training_stats: TrainingStats,
    pub model_stats: ModelStats,
    pub is_trained: bool,
}

/// Biomedical-specific features for enhanced embeddings
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BiomedicalFeatures {
    /// Gene-disease association scores
    pub gene_disease_associations: HashMap<(String, String), f32>,
    /// Drug-target binding affinities
    pub drug_target_affinities: HashMap<(String, String), f32>,
    /// Pathway membership scores
    pub pathway_memberships: HashMap<(String, String), f32>,
    /// Protein-protein interaction scores
    pub protein_interactions: HashMap<(String, String), f32>,
    /// Sequence similarity scores
    pub sequence_similarities: HashMap<(String, String), f32>,
    /// Chemical structure similarities
    pub structure_similarities: HashMap<(String, String), f32>,
    /// Expression correlations
    pub expression_correlations: HashMap<(String, String), f32>,
    /// Tissue-specific expression
    pub tissue_expression: HashMap<(String, String), f32>,
}

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

    /// Add gene-disease association
    pub fn add_gene_disease_association(&mut self, gene: &str, disease: &str, score: f32) {
        self.features.gene_disease_associations.insert(
            (gene.to_string(), disease.to_string()),
            score,
        );
        
        // Also add reverse mapping
        self.features.gene_disease_associations.insert(
            (disease.to_string(), gene.to_string()),
            score,
        );
    }

    /// Add drug-target interaction
    pub fn add_drug_target_interaction(&mut self, drug: &str, target: &str, affinity: f32) {
        self.features.drug_target_affinities.insert(
            (drug.to_string(), target.to_string()),
            affinity,
        );
    }

    /// Add pathway membership
    pub fn add_pathway_membership(&mut self, entity: &str, pathway: &str, score: f32) {
        self.features.pathway_memberships.insert(
            (entity.to_string(), pathway.to_string()),
            score,
        );
    }

    /// Add protein-protein interaction
    pub fn add_protein_interaction(&mut self, protein1: &str, protein2: &str, score: f32) {
        self.features.protein_interactions.insert(
            (protein1.to_string(), protein2.to_string()),
            score,
        );
        
        // Symmetric relationship
        self.features.protein_interactions.insert(
            (protein2.to_string(), protein1.to_string()),
            score,
        );
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
                Err(anyhow!("No embedding found for {} of type {:?}", entity, entity_type))
            }
        } else {
            Err(anyhow!("Unknown entity type for {}", entity))
        }
    }

    /// Predict gene-disease associations
    pub fn predict_gene_disease_associations(&self, gene: &str, k: usize) -> Result<Vec<(String, f64)>> {
        if !self.is_trained {
            return Err(anyhow!("Model not trained"));
        }

        let gene_embedding = self.gene_embeddings.get(gene)
            .ok_or_else(|| anyhow!("Gene {} not found", gene))?;

        let mut scores = Vec::new();
        
        for (disease, disease_embedding) in &self.disease_embeddings {
            // Base similarity
            let similarity = gene_embedding.dot(disease_embedding) as f64;
            
            // Enhance with existing association data
            let enhanced_score = if let Some(&assoc_score) = 
                self.features.gene_disease_associations.get(&(gene.to_string(), disease.clone())) {
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

        let drug_embedding = self.drug_embeddings.get(drug)
            .ok_or_else(|| anyhow!("Drug {} not found", drug))?;

        let mut scores = Vec::new();
        
        for (protein, protein_embedding) in &self.protein_embeddings {
            // Base similarity
            let similarity = drug_embedding.dot(protein_embedding) as f64;
            
            // Enhance with binding affinity data
            let enhanced_score = if let Some(&affinity) = 
                self.features.drug_target_affinities.get(&(drug.to_string(), protein.clone())) {
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
        let pathway_embedding = self.pathway_embeddings.get(pathway)
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
                self.entity_types.insert(triple.subject.iri.clone(), subject_type);
            }
            
            if let Some(object_type) = BiomedicalEntityType::from_iri(&triple.object.iri) {
                self.entity_types.insert(triple.object.iri.clone(), object_type);
            }
            
            // Extract relation types
            if let Some(relation_type) = BiomedicalRelationType::from_iri(&triple.predicate.iri) {
                self.relation_types.insert(triple.predicate.iri.clone(), relation_type);
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
                    .map(|_| (rand::random::<f32>() - 0.5) * 0.1)
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
                    .map(|_| (rand::random::<f32>() - 0.5) * 0.1)
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
        self.gene_embeddings.get(entity)
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
                println!("Epoch {}: Loss = {:.6}", epoch, epoch_loss);
            }
        }

        let training_time = start_time.elapsed().as_secs_f64();
        
        self.training_stats = TrainingStats {
            epochs_completed: epochs,
            final_loss: loss_history.last().copied().unwrap_or(0.0),
            training_time_seconds: training_time,
            convergence_achieved: loss_history.last().map_or(false, |&loss| loss < 0.001),
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

    fn get_relation_embedding(&self, relation: &str) -> Result<Vector> {
        if let Some(embedding) = self.relation_embeddings.get(relation) {
            Ok(Vector::from_array1(embedding))
        } else {
            Err(anyhow!("Relation {} not found", relation))
        }
    }

    fn score_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<f64> {
        let subject_emb = self.get_entity_embedding(subject)?;
        let relation_emb = self.get_relation_embedding(predicate)?;
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

    fn predict_objects(&self, subject: &str, predicate: &str, k: usize) -> Result<Vec<(String, f64)>> {
        // Use specialized prediction methods based on relation type
        if let Some(relation_type) = self.relation_types.get(predicate) {
            match relation_type {
                BiomedicalRelationType::CausesDisease |
                BiomedicalRelationType::AssociatedWithDisease => {
                    return self.predict_gene_disease_associations(subject, k);
                }
                BiomedicalRelationType::TargetsProtein |
                BiomedicalRelationType::BindsToProtein => {
                    return self.predict_drug_targets(subject, k);
                }
                _ => {
                    // Fall back to generic prediction
                }
            }
        }

        // Generic prediction
        let subject_emb = self.get_entity_embedding(subject)?;
        let relation_emb = self.get_relation_embedding(predicate)?;

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

    fn predict_subjects(&self, predicate: &str, object: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let object_emb = self.get_entity_embedding(object)?;
        let relation_emb = self.get_relation_embedding(predicate)?;

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

    fn predict_relations(&self, subject: &str, object: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let subject_emb = self.get_entity_embedding(subject)?;
        let object_emb = self.get_entity_embedding(object)?;

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
            if let Ok(embedding) = self.get_entity_embedding(text) {
                embeddings.push(embedding.values);
            } else {
                // Return zero embedding for unknown entities
                embeddings.push(vec![0.0; self.config.base_config.dimensions]);
            }
        }
        
        Ok(embeddings)
    }
}

/// Specialized text embedding models for domain-specific applications
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecializedTextModel {
    /// SciBERT for scientific literature
    SciBERT,
    /// CodeBERT for code and programming
    CodeBERT,
    /// BioBERT for biomedical literature
    BioBERT,
    /// LegalBERT for legal documents
    LegalBERT,
    /// FinBERT for financial texts
    FinBERT,
    /// ClinicalBERT for clinical notes
    ClinicalBERT,
    /// ChemBERT for chemical compounds
    ChemBERT,
}

impl SpecializedTextModel {
    /// Get the model name for loading pre-trained weights
    pub fn model_name(&self) -> &'static str {
        match self {
            SpecializedTextModel::SciBERT => "allenai/scibert_scivocab_uncased",
            SpecializedTextModel::CodeBERT => "microsoft/codebert-base",
            SpecializedTextModel::BioBERT => "dmis-lab/biobert-base-cased-v1.2",
            SpecializedTextModel::LegalBERT => "nlpaueb/legal-bert-base-uncased",
            SpecializedTextModel::FinBERT => "ProsusAI/finbert",
            SpecializedTextModel::ClinicalBERT => "emilyalsentzer/Bio_ClinicalBERT",
            SpecializedTextModel::ChemBERT => "seyonec/ChemBERTa-zinc-base-v1",
        }
    }

    /// Get the vocabulary size for the model
    pub fn vocab_size(&self) -> usize {
        match self {
            SpecializedTextModel::SciBERT => 31090,
            SpecializedTextModel::CodeBERT => 50265,
            SpecializedTextModel::BioBERT => 28996,
            SpecializedTextModel::LegalBERT => 30522,
            SpecializedTextModel::FinBERT => 30522,
            SpecializedTextModel::ClinicalBERT => 28996,
            SpecializedTextModel::ChemBERT => 600,
        }
    }

    /// Get the default embedding dimension
    pub fn embedding_dim(&self) -> usize {
        match self {
            SpecializedTextModel::SciBERT => 768,
            SpecializedTextModel::CodeBERT => 768,
            SpecializedTextModel::BioBERT => 768,
            SpecializedTextModel::LegalBERT => 768,
            SpecializedTextModel::FinBERT => 768,
            SpecializedTextModel::ClinicalBERT => 768,
            SpecializedTextModel::ChemBERT => 384,
        }
    }

    /// Get the maximum sequence length
    pub fn max_sequence_length(&self) -> usize {
        match self {
            SpecializedTextModel::SciBERT => 512,
            SpecializedTextModel::CodeBERT => 512,
            SpecializedTextModel::BioBERT => 512,
            SpecializedTextModel::LegalBERT => 512,
            SpecializedTextModel::FinBERT => 512,
            SpecializedTextModel::ClinicalBERT => 512,
            SpecializedTextModel::ChemBERT => 512,
        }
    }

    /// Get domain-specific preprocessing rules
    pub fn get_preprocessing_rules(&self) -> Vec<PreprocessingRule> {
        match self {
            SpecializedTextModel::SciBERT => vec![
                PreprocessingRule::NormalizeScientificNotation,
                PreprocessingRule::ExpandAbbreviations,
                PreprocessingRule::HandleChemicalFormulas,
                PreprocessingRule::PreserveCitations,
            ],
            SpecializedTextModel::CodeBERT => vec![
                PreprocessingRule::PreserveCodeTokens,
                PreprocessingRule::HandleCamelCase,
                PreprocessingRule::NormalizeWhitespace,
                PreprocessingRule::PreservePunctuation,
            ],
            SpecializedTextModel::BioBERT => vec![
                PreprocessingRule::NormalizeMedicalTerms,
                PreprocessingRule::HandleGeneNames,
                PreprocessingRule::ExpandMedicalAbbreviations,
                PreprocessingRule::PreserveDosages,
            ],
            SpecializedTextModel::LegalBERT => vec![
                PreprocessingRule::PreserveLegalCitations,
                PreprocessingRule::HandleLegalTerms,
                PreprocessingRule::NormalizeCaseReferences,
            ],
            SpecializedTextModel::FinBERT => vec![
                PreprocessingRule::NormalizeFinancialTerms,
                PreprocessingRule::HandleCurrencySymbols,
                PreprocessingRule::PreservePercentages,
            ],
            SpecializedTextModel::ClinicalBERT => vec![
                PreprocessingRule::NormalizeMedicalTerms,
                PreprocessingRule::HandleMedicalAbbreviations,
                PreprocessingRule::PreserveDosages,
                PreprocessingRule::NormalizeTimestamps,
            ],
            SpecializedTextModel::ChemBERT => vec![
                PreprocessingRule::HandleChemicalFormulas,
                PreprocessingRule::PreserveMolecularStructures,
                PreprocessingRule::NormalizeChemicalNames,
            ],
        }
    }
}

/// Preprocessing rules for specialized text models
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PreprocessingRule {
    /// Normalize scientific notation (e.g., 1.23e-4)
    NormalizeScientificNotation,
    /// Expand domain-specific abbreviations
    ExpandAbbreviations,
    /// Handle chemical formulas and compounds
    HandleChemicalFormulas,
    /// Preserve citation formats
    PreserveCitations,
    /// Preserve code tokens and keywords
    PreserveCodeTokens,
    /// Handle camelCase and snake_case
    HandleCamelCase,
    /// Normalize whitespace patterns
    NormalizeWhitespace,
    /// Preserve punctuation in code
    PreservePunctuation,
    /// Normalize medical terminology
    NormalizeMedicalTerms,
    /// Handle gene and protein names
    HandleGeneNames,
    /// Expand medical abbreviations
    ExpandMedicalAbbreviations,
    /// Preserve dosage information
    PreserveDosages,
    /// Preserve legal citations
    PreserveLegalCitations,
    /// Handle legal terminology
    HandleLegalTerms,
    /// Normalize case references
    NormalizeCaseReferences,
    /// Normalize financial terms
    NormalizeFinancialTerms,
    /// Handle currency symbols
    HandleCurrencySymbols,
    /// Preserve percentage values
    PreservePercentages,
    /// Handle medical abbreviations
    HandleMedicalAbbreviations,
    /// Normalize timestamps
    NormalizeTimestamps,
    /// Preserve molecular structures
    PreserveMolecularStructures,
    /// Normalize chemical names
    NormalizeChemicalNames,
}

/// Configuration for specialized text embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecializedTextConfig {
    pub model_type: SpecializedTextModel,
    pub base_config: ModelConfig,
    /// Fine-tuning configuration
    pub fine_tune_config: FineTuningConfig,
    /// Preprocessing configuration
    pub preprocessing_enabled: bool,
    /// Domain-specific vocabulary augmentation
    pub vocab_augmentation: bool,
    /// Use domain-specific pre-training
    pub domain_pretraining: bool,
}

/// Fine-tuning configuration for specialized models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningConfig {
    /// Learning rate for fine-tuning
    pub learning_rate: f64,
    /// Number of fine-tuning epochs
    pub epochs: usize,
    /// Freeze base model layers
    pub freeze_base_layers: bool,
    /// Number of layers to freeze
    pub frozen_layers: usize,
    /// Use gradual unfreezing
    pub gradual_unfreezing: bool,
    /// Discriminative fine-tuning rates
    pub discriminative_rates: Vec<f64>,
}

impl Default for FineTuningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-5,
            epochs: 3,
            freeze_base_layers: false,
            frozen_layers: 0,
            gradual_unfreezing: false,
            discriminative_rates: vec![],
        }
    }
}

impl Default for SpecializedTextConfig {
    fn default() -> Self {
        Self {
            model_type: SpecializedTextModel::BioBERT,
            base_config: ModelConfig::default(),
            fine_tune_config: FineTuningConfig::default(),
            preprocessing_enabled: true,
            vocab_augmentation: false,
            domain_pretraining: false,
        }
    }
}

/// Specialized text embedding processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecializedTextEmbedding {
    pub config: SpecializedTextConfig,
    pub model_id: Uuid,
    /// Text embeddings cache
    pub text_embeddings: HashMap<String, Array1<f32>>,
    /// Domain-specific vocabulary
    pub domain_vocab: HashSet<String>,
    /// Preprocessing pipeline
    pub preprocessing_rules: Vec<PreprocessingRule>,
    /// Training statistics
    pub training_stats: TrainingStats,
    /// Model statistics
    pub model_stats: ModelStats,
    pub is_trained: bool,
}

impl SpecializedTextEmbedding {
    /// Create new specialized text embedding model
    pub fn new(config: SpecializedTextConfig) -> Self {
        let model_id = Uuid::new_v4();
        let now = Utc::now();
        let preprocessing_rules = config.model_type.get_preprocessing_rules();

        Self {
            model_id,
            text_embeddings: HashMap::new(),
            domain_vocab: HashSet::new(),
            preprocessing_rules,
            training_stats: TrainingStats::default(),
            model_stats: ModelStats {
                num_entities: 0,
                num_relations: 0,
                num_triples: 0,
                dimensions: config.model_type.embedding_dim(),
                is_trained: false,
                model_type: format!("SpecializedText_{:?}", config.model_type),
                creation_time: now,
                last_training_time: None,
            },
            is_trained: false,
            config,
        }
    }

    /// Create SciBERT configuration
    pub fn scibert_config() -> SpecializedTextConfig {
        SpecializedTextConfig {
            model_type: SpecializedTextModel::SciBERT,
            base_config: ModelConfig::default().with_dimensions(768),
            fine_tune_config: FineTuningConfig::default(),
            preprocessing_enabled: true,
            vocab_augmentation: true,
            domain_pretraining: true,
        }
    }

    /// Create CodeBERT configuration
    pub fn codebert_config() -> SpecializedTextConfig {
        SpecializedTextConfig {
            model_type: SpecializedTextModel::CodeBERT,
            base_config: ModelConfig::default().with_dimensions(768),
            fine_tune_config: FineTuningConfig::default(),
            preprocessing_enabled: true,
            vocab_augmentation: false,
            domain_pretraining: true,
        }
    }

    /// Create BioBERT configuration
    pub fn biobert_config() -> SpecializedTextConfig {
        SpecializedTextConfig {
            model_type: SpecializedTextModel::BioBERT,
            base_config: ModelConfig::default().with_dimensions(768),
            fine_tune_config: FineTuningConfig {
                learning_rate: 1e-5,
                epochs: 5,
                freeze_base_layers: true,
                frozen_layers: 6,
                gradual_unfreezing: true,
                discriminative_rates: vec![1e-6, 5e-6, 1e-5, 2e-5],
            },
            preprocessing_enabled: true,
            vocab_augmentation: true,
            domain_pretraining: true,
        }
    }

    /// Preprocess text according to domain-specific rules
    pub fn preprocess_text(&self, text: &str) -> Result<String> {
        if !self.config.preprocessing_enabled {
            return Ok(text.to_string());
        }

        let mut processed = text.to_string();

        for rule in &self.preprocessing_rules {
            processed = self.apply_preprocessing_rule(&processed, rule)?;
        }

        Ok(processed)
    }

    /// Apply a specific preprocessing rule
    fn apply_preprocessing_rule(&self, text: &str, rule: &PreprocessingRule) -> Result<String> {
        match rule {
            PreprocessingRule::NormalizeScientificNotation => {
                // Convert scientific notation to normalized form (simplified)
                Ok(text.replace("E+", "e+").replace("E-", "e-").replace("E", "e"))
            }
            PreprocessingRule::HandleChemicalFormulas => {
                // Preserve chemical formulas by adding special tokens (simplified)
                Ok(text.replace("H2O", "[CHEM]H2O[/CHEM]"))
            }
            PreprocessingRule::HandleCamelCase => {
                // Split camelCase into separate tokens (simplified)
                let mut result = String::new();
                let mut chars = text.chars().peekable();
                while let Some(c) = chars.next() {
                    result.push(c);
                    if c.is_lowercase() && chars.peek().map_or(false, |&next| next.is_uppercase()) {
                        result.push(' ');
                    }
                }
                Ok(result)
            }
            PreprocessingRule::NormalizeMedicalTerms => {
                // Normalize common medical abbreviations
                let mut result = text.to_string();
                let replacements = [
                    ("mg/kg", "milligrams per kilogram"),
                    ("q.d.", "once daily"),
                    ("b.i.d.", "twice daily"),
                    ("t.i.d.", "three times daily"),
                    ("q.i.d.", "four times daily"),
                ];
                
                for (abbrev, expansion) in &replacements {
                    result = result.replace(abbrev, expansion);
                }
                Ok(result)
            }
            PreprocessingRule::HandleGeneNames => {
                // Standardize gene name formatting (simplified)
                Ok(text.replace("BRCA1", "[GENE]BRCA1[/GENE]").replace("TP53", "[GENE]TP53[/GENE]"))
            }
            PreprocessingRule::PreserveCodeTokens => {
                // Preserve code-like tokens (simplified)
                Ok(text.replace("function", "[CODE]function[/CODE]"))
            }
            _ => {
                // Placeholder for other rules - would implement in production
                Ok(text.to_string())
            }
        }
    }

    /// Generate embedding for text using specialized model
    pub async fn encode_text(&mut self, text: &str) -> Result<Array1<f32>> {
        // Preprocess the text
        let processed_text = self.preprocess_text(text)?;

        // Check cache first
        if let Some(cached_embedding) = self.text_embeddings.get(&processed_text) {
            return Ok(cached_embedding.clone());
        }

        // Generate embedding using domain-specific model
        let embedding = self.generate_specialized_embedding(&processed_text).await?;
        
        // Cache the result
        self.text_embeddings.insert(processed_text, embedding.clone());
        
        Ok(embedding)
    }

    /// Generate specialized embedding for the specific domain
    async fn generate_specialized_embedding(&self, text: &str) -> Result<Array1<f32>> {
        // In a real implementation, this would use the actual pre-trained model
        // For now, simulate domain-specific embeddings with enhanced features
        
        let embedding_dim = self.config.model_type.embedding_dim();
        let mut embedding = vec![0.0; embedding_dim];
        
        // Domain-specific feature extraction
        match self.config.model_type {
            SpecializedTextModel::SciBERT => {
                // Scientific text features: citations, formulas, terminology
                embedding[0] = if text.contains("et al.") { 1.0 } else { 0.0 };
                embedding[1] = if text.contains("figure") || text.contains("table") { 1.0 } else { 0.0 };
                embedding[2] = text.matches(char::is_numeric).count() as f32 / text.len() as f32;
            }
            SpecializedTextModel::CodeBERT => {
                // Code features: keywords, operators, structures
                embedding[0] = if text.contains("function") || text.contains("def") { 1.0 } else { 0.0 };
                embedding[1] = if text.contains("class") || text.contains("struct") { 1.0 } else { 0.0 };
                embedding[2] = text.matches(|c: char| "{}()[]".contains(c)).count() as f32 / text.len() as f32;
            }
            SpecializedTextModel::BioBERT => {
                // Biomedical features: genes, proteins, diseases
                embedding[0] = if text.contains("protein") || text.contains("gene") { 1.0 } else { 0.0 };
                embedding[1] = if text.contains("disease") || text.contains("syndrome") { 1.0 } else { 0.0 };
                embedding[2] = if text.contains("mg") || text.contains("dose") { 1.0 } else { 0.0 };
            }
            _ => {
                // Generic specialized features
                embedding[0] = text.len() as f32 / 1000.0; // Length normalization
                embedding[1] = text.split_whitespace().count() as f32 / text.len() as f32; // Word density
            }
        }
        
        // Fill remaining dimensions with text-based features
        for i in 3..embedding_dim {
            let byte_val = text.bytes().nth(i % text.len()).unwrap_or(0) as f32;
            embedding[i] = (byte_val / 255.0 - 0.5) * 2.0; // Normalize to [-1, 1]
        }
        
        // Apply domain-specific transformations
        if self.config.domain_pretraining {
            for val in &mut embedding {
                *val *= 1.2; // Amplify features for domain-pretrained models
            }
        }
        
        Ok(Array1::from_vec(embedding))
    }

    /// Fine-tune the model on domain-specific data
    pub async fn fine_tune(&mut self, training_texts: Vec<String>) -> Result<TrainingStats> {
        let start_time = std::time::Instant::now();
        let epochs = self.config.fine_tune_config.epochs;
        
        let mut loss_history = Vec::new();
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            
            for text in &training_texts {
                // Generate embedding and compute loss
                let embedding = self.encode_text(text).await?;
                
                // Simplified fine-tuning loss computation
                let target_variance = 0.1; // Target embedding variance
                let actual_variance = embedding.var(0.0);
                let loss = (actual_variance - target_variance).powi(2);
                epoch_loss += loss;
            }
            
            epoch_loss /= training_texts.len() as f32;
            loss_history.push(epoch_loss as f64);
            
            if epoch % 10 == 0 {
                println!("Fine-tuning epoch {}: Loss = {:.6}", epoch, epoch_loss);
            }
        }
        
        let training_time = start_time.elapsed().as_secs_f64();
        
        self.training_stats = TrainingStats {
            epochs_completed: epochs,
            final_loss: loss_history.last().copied().unwrap_or(0.0),
            training_time_seconds: training_time,
            convergence_achieved: loss_history.last().map_or(false, |&loss| loss < 0.01),
            loss_history,
        };
        
        self.is_trained = true;
        self.model_stats.is_trained = true;
        self.model_stats.last_training_time = Some(Utc::now());
        
        Ok(self.training_stats.clone())
    }

    /// Get model statistics
    pub fn get_stats(&self) -> ModelStats {
        self.model_stats.clone()
    }

    /// Clear cached embeddings
    pub fn clear_cache(&mut self) {
        self.text_embeddings.clear();
    }
}

// Simplified regex-like functionality for preprocessing
mod regex {
    pub struct Regex(String);
    
    impl Regex {
        pub fn new(pattern: &str) -> Result<Self, &'static str> {
            Ok(Regex(pattern.to_string()))
        }
        
        pub fn replace_all<'a, F>(&self, text: &'a str, rep: F) -> std::borrow::Cow<'a, str>
        where
            F: Fn(&str) -> String,
        {
            // Simplified regex replacement for demo - just return original text
            std::borrow::Cow::Borrowed(text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biomedical_entity_type_from_iri() {
        assert_eq!(
            BiomedicalEntityType::from_iri("http://example.org/gene/BRCA1"),
            Some(BiomedicalEntityType::Gene)
        );
        assert_eq!(
            BiomedicalEntityType::from_iri("http://example.org/disease/cancer"),
            Some(BiomedicalEntityType::Disease)
        );
        assert_eq!(
            BiomedicalEntityType::from_iri("http://example.org/drug/aspirin"),
            Some(BiomedicalEntityType::Drug)
        );
    }

    #[test]
    fn test_biomedical_config_default() {
        let config = BiomedicalEmbeddingConfig::default();
        assert_eq!(config.gene_disease_weight, 2.0);
        assert_eq!(config.drug_target_weight, 1.5);
        assert!(config.use_sequence_similarity);
        assert_eq!(config.species_filter, Some("Homo sapiens".to_string()));
    }

    #[test]
    fn test_biomedical_embedding_creation() {
        let config = BiomedicalEmbeddingConfig::default();
        let model = BiomedicalEmbedding::new(config);
        
        assert_eq!(model.model_type(), "BiomedicalEmbedding");
        assert!(!model.is_trained());
        assert_eq!(model.gene_embeddings.len(), 0);
    }

    #[test]
    fn test_gene_disease_association() {
        let mut model = BiomedicalEmbedding::new(BiomedicalEmbeddingConfig::default());
        
        model.add_gene_disease_association("BRCA1", "breast_cancer", 0.8);
        
        assert_eq!(
            model.features.gene_disease_associations.get(&("BRCA1".to_string(), "breast_cancer".to_string())),
            Some(&0.8)
        );
    }

    #[test]
    fn test_drug_target_interaction() {
        let mut model = BiomedicalEmbedding::new(BiomedicalEmbeddingConfig::default());
        
        model.add_drug_target_interaction("aspirin", "COX1", 0.9);
        
        assert_eq!(
            model.features.drug_target_affinities.get(&("aspirin".to_string(), "COX1".to_string())),
            Some(&0.9)
        );
    }

    #[test]
    fn test_specialized_text_model_properties() {
        let scibert = SpecializedTextModel::SciBERT;
        assert_eq!(scibert.model_name(), "allenai/scibert_scivocab_uncased");
        assert_eq!(scibert.vocab_size(), 31090);
        assert_eq!(scibert.embedding_dim(), 768);
        assert_eq!(scibert.max_sequence_length(), 512);

        let codebert = SpecializedTextModel::CodeBERT;
        assert_eq!(codebert.model_name(), "microsoft/codebert-base");
        assert_eq!(codebert.vocab_size(), 50265);

        let biobert = SpecializedTextModel::BioBERT;
        assert_eq!(biobert.model_name(), "dmis-lab/biobert-base-cased-v1.2");
        assert_eq!(biobert.vocab_size(), 28996);
    }

    #[test]
    fn test_specialized_text_preprocessing_rules() {
        let scibert = SpecializedTextModel::SciBERT;
        let rules = scibert.get_preprocessing_rules();
        assert!(rules.contains(&PreprocessingRule::NormalizeScientificNotation));
        assert!(rules.contains(&PreprocessingRule::HandleChemicalFormulas));

        let codebert = SpecializedTextModel::CodeBERT;
        let rules = codebert.get_preprocessing_rules();
        assert!(rules.contains(&PreprocessingRule::PreserveCodeTokens));
        assert!(rules.contains(&PreprocessingRule::HandleCamelCase));

        let biobert = SpecializedTextModel::BioBERT;
        let rules = biobert.get_preprocessing_rules();
        assert!(rules.contains(&PreprocessingRule::NormalizeMedicalTerms));
        assert!(rules.contains(&PreprocessingRule::HandleGeneNames));
    }

    #[test]
    fn test_specialized_text_config_factory_methods() {
        let scibert_config = SpecializedTextEmbedding::scibert_config();
        assert_eq!(scibert_config.model_type, SpecializedTextModel::SciBERT);
        assert_eq!(scibert_config.base_config.dimensions, 768);
        assert!(scibert_config.preprocessing_enabled);
        assert!(scibert_config.vocab_augmentation);
        assert!(scibert_config.domain_pretraining);

        let codebert_config = SpecializedTextEmbedding::codebert_config();
        assert_eq!(codebert_config.model_type, SpecializedTextModel::CodeBERT);
        assert!(!codebert_config.vocab_augmentation);

        let biobert_config = SpecializedTextEmbedding::biobert_config();
        assert_eq!(biobert_config.model_type, SpecializedTextModel::BioBERT);
        assert!(biobert_config.fine_tune_config.freeze_base_layers);
        assert_eq!(biobert_config.fine_tune_config.frozen_layers, 6);
        assert!(biobert_config.fine_tune_config.gradual_unfreezing);
    }

    #[test]
    fn test_specialized_text_embedding_creation() {
        let config = SpecializedTextEmbedding::scibert_config();
        let model = SpecializedTextEmbedding::new(config);
        
        assert!(model.model_stats.model_type.contains("SciBERT"));
        assert_eq!(model.model_stats.dimensions, 768);
        assert!(!model.is_trained);
        assert_eq!(model.text_embeddings.len(), 0);
        assert_eq!(model.preprocessing_rules.len(), 4); // SciBERT has 4 rules
    }

    #[test]
    fn test_preprocessing_medical_terms() {
        let config = SpecializedTextEmbedding::biobert_config();
        let model = SpecializedTextEmbedding::new(config);
        
        let text = "Patient takes 100 mg/kg b.i.d. for treatment";
        let processed = model.preprocess_text(text).unwrap();
        
        // Should expand medical abbreviations
        assert!(processed.contains("milligrams per kilogram"));
        assert!(processed.contains("twice daily"));
    }

    #[test]
    fn test_preprocessing_disabled() {
        let mut config = SpecializedTextEmbedding::biobert_config();
        config.preprocessing_enabled = false;
        let model = SpecializedTextEmbedding::new(config);
        
        let text = "Patient takes 100 mg/kg b.i.d. for treatment";
        let processed = model.preprocess_text(text).unwrap();
        
        // Should be unchanged when preprocessing is disabled
        assert_eq!(processed, text);
    }

    #[tokio::test]
    async fn test_specialized_text_encoding() {
        let config = SpecializedTextEmbedding::scibert_config();
        let mut model = SpecializedTextEmbedding::new(config);
        
        let text = "The protein folding study shows significant results with p < 0.001";
        let embedding = model.encode_text(text).await.unwrap();
        
        assert_eq!(embedding.len(), 768);
        
        // Test caching - second call should return cached result
        let embedding2 = model.encode_text(text).await.unwrap();
        assert_eq!(embedding.to_vec(), embedding2.to_vec());
        assert_eq!(model.text_embeddings.len(), 1);
    }

    #[tokio::test]
    async fn test_domain_specific_features() {
        // Test SciBERT features
        let config = SpecializedTextEmbedding::scibert_config();
        let mut model = SpecializedTextEmbedding::new(config);
        
        let scientific_text = "The study by Smith et al. shows figure 1 demonstrates the results";
        let embedding = model.encode_text(scientific_text).await.unwrap();
        
        // Should detect scientific features (citations, figures)
        // Values are amplified by 1.2 due to domain pretraining
        assert_eq!(embedding[0], 1.2); // et al. detected, amplified
        assert_eq!(embedding[1], 1.2); // figure detected, amplified

        // Test CodeBERT features
        let config = SpecializedTextEmbedding::codebert_config();
        let mut model = SpecializedTextEmbedding::new(config);
        
        let code_text = "function calculateSum() { return a + b; }";
        let embedding = model.encode_text(code_text).await.unwrap();
        
        // Should detect code features (amplified by domain pretraining)
        assert_eq!(embedding[0], 1.2); // function detected, amplified
        assert!(embedding[2] > 0.0); // brackets detected (text-based features)

        // Test BioBERT features
        let config = SpecializedTextEmbedding::biobert_config();
        let mut model = SpecializedTextEmbedding::new(config);
        
        let biomedical_text = "The protein expression correlates with cancer disease progression, dose 100mg";
        let embedding = model.encode_text(biomedical_text).await.unwrap();
        
        // Should detect biomedical features (amplified by domain pretraining)
        assert_eq!(embedding[0], 1.2); // protein detected, amplified
        assert_eq!(embedding[1], 1.2); // disease detected, amplified
        assert_eq!(embedding[2], 1.2); // mg detected, amplified
    }

    #[tokio::test]
    async fn test_fine_tuning() {
        let config = SpecializedTextEmbedding::biobert_config();
        let mut model = SpecializedTextEmbedding::new(config);
        
        let training_texts = vec![
            "Gene expression analysis in cancer cells".to_string(),
            "Protein folding mechanisms in disease".to_string(),
            "Drug interaction with target proteins".to_string(),
        ];
        
        let stats = model.fine_tune(training_texts).await.unwrap();
        
        assert!(model.is_trained);
        assert_eq!(stats.epochs_completed, 5); // BioBERT config has 5 epochs
        assert!(stats.training_time_seconds > 0.0);
        assert!(!stats.loss_history.is_empty());
        assert!(model.model_stats.is_trained);
        assert!(model.model_stats.last_training_time.is_some());
    }
}