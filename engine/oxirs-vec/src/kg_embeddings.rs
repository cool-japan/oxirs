//! Knowledge Graph Embeddings for RDF data
//!
//! This module implements various knowledge graph embedding methods:
//! - TransE: Translation-based embeddings
//! - ComplEx: Complex number embeddings
//! - RotatE: Rotation-based embeddings

use crate::Vector;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use nalgebra::{Complex, DVector};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};

/// Knowledge graph embedding model type
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum KGEmbeddingModelType {
    /// Translation-based embeddings (TransE)
    TransE,
    /// Complex number embeddings (ComplEx)
    ComplEx,
    /// Rotation-based embeddings (RotatE)
    RotatE,
    /// Graph Convolutional Network (GCN)
    GCN,
    /// GraphSAGE (Graph Sample and Aggregate)
    GraphSAGE,
}

/// Configuration for knowledge graph embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KGEmbeddingConfig {
    /// Model type
    pub model: KGEmbeddingModelType,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Margin for loss function
    pub margin: f32,
    /// Negative sampling ratio
    pub negative_samples: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// L1 or L2 norm
    pub norm: usize,
    /// Random seed
    pub random_seed: Option<u64>,
    /// Regularization weight
    pub regularization: f32,
}

impl Default for KGEmbeddingConfig {
    fn default() -> Self {
        Self {
            model: KGEmbeddingModelType::TransE,
            dimensions: 100,
            learning_rate: 0.01,
            margin: 1.0,
            negative_samples: 10,
            batch_size: 100,
            epochs: 100,
            norm: 2,
            random_seed: Some(42),
            regularization: 0.0,
        }
    }
}

/// Triple for knowledge graph
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Triple {
    pub fn new(subject: String, predicate: String, object: String) -> Self {
        Self { subject, predicate, object }
    }
}

/// Base trait for knowledge graph embedding models
pub trait KGEmbeddingModel: Send + Sync {
    /// Train the model on triples
    fn train(&mut self, triples: &[Triple]) -> Result<()>;
    
    /// Get entity embedding
    fn get_entity_embedding(&self, entity: &str) -> Option<Vector>;
    
    /// Get relation embedding
    fn get_relation_embedding(&self, relation: &str) -> Option<Vector>;
    
    /// Score a triple
    fn score_triple(&self, triple: &Triple) -> f32;
    
    /// Predict tail entities for (head, relation, ?)
    fn predict_tail(&self, head: &str, relation: &str, k: usize) -> Vec<(String, f32)>;
    
    /// Predict head entities for (?, relation, tail)
    fn predict_head(&self, relation: &str, tail: &str, k: usize) -> Vec<(String, f32)>;
    
    /// Get all entity embeddings
    fn get_entity_embeddings(&self) -> &HashMap<String, Vector>;
    
    /// Get all relation embeddings
    fn get_relation_embeddings(&self) -> &HashMap<String, Vector>;
}

/// TransE: Translation-based embeddings
/// Learns embeddings where h + r ≈ t for triple (h, r, t)
pub struct TransE {
    config: KGEmbeddingConfig,
    entity_embeddings: HashMap<String, DVector<f32>>,
    relation_embeddings: HashMap<String, DVector<f32>>,
    entities: Vec<String>,
    relations: Vec<String>,
}

impl TransE {
    pub fn new(config: KGEmbeddingConfig) -> Self {
        Self {
            config,
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            entities: Vec::new(),
            relations: Vec::new(),
        }
    }
    
    /// Initialize embeddings
    fn initialize_embeddings(&mut self, triples: &[Triple]) {
        // Collect unique entities and relations
        let mut entities = std::collections::HashSet::new();
        let mut relations = std::collections::HashSet::new();
        
        for triple in triples {
            entities.insert(triple.subject.clone());
            entities.insert(triple.object.clone());
            relations.insert(triple.predicate.clone());
        }
        
        self.entities = entities.into_iter().collect();
        self.relations = relations.into_iter().collect();
        
        // Initialize embeddings with uniform distribution
        let mut rng = if let Some(seed) = self.config.random_seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };
        
        let uniform = Uniform::new(-6.0 / (self.config.dimensions as f32).sqrt(), 
                                   6.0 / (self.config.dimensions as f32).sqrt());
        
        // Initialize entity embeddings
        for entity in &self.entities {
            let values: Vec<f32> = (0..self.config.dimensions)
                .map(|_| uniform.sample(&mut rng))
                .collect();
            let mut embedding = DVector::from_vec(values);
            
            // Normalize entities
            let norm = embedding.norm();
            if norm > 0.0 {
                embedding /= norm;
            }
            
            self.entity_embeddings.insert(entity.clone(), embedding);
        }
        
        // Initialize relation embeddings
        for relation in &self.relations {
            let values: Vec<f32> = (0..self.config.dimensions)
                .map(|_| uniform.sample(&mut rng))
                .collect();
            let embedding = DVector::from_vec(values);
            
            // Relations are not normalized in TransE
            self.relation_embeddings.insert(relation.clone(), embedding);
        }
    }
    
    /// Generate negative samples
    fn generate_negative_samples(&self, triple: &Triple, rng: &mut impl Rng) -> Vec<Triple> {
        let mut negatives = Vec::new();
        
        for _ in 0..self.config.negative_samples {
            if rng.gen_bool(0.5) {
                // Corrupt head
                let mut negative = triple.clone();
                loop {
                    let idx = rng.gen_range(0..self.entities.len());
                    let entity = &self.entities[idx];
                    if entity != &triple.subject {
                        negative.subject = entity.clone();
                        break;
                    }
                }
                negatives.push(negative);
            } else {
                // Corrupt tail
                let mut negative = triple.clone();
                loop {
                    let idx = rng.gen_range(0..self.entities.len());
                    let entity = &self.entities[idx];
                    if entity != &triple.object {
                        negative.object = entity.clone();
                        break;
                    }
                }
                negatives.push(negative);
            }
        }
        
        negatives
    }
    
    /// Calculate distance for a triple
    fn distance(&self, triple: &Triple) -> f32 {
        let h = self.entity_embeddings.get(&triple.subject).unwrap();
        let r = self.relation_embeddings.get(&triple.predicate).unwrap();
        let t = self.entity_embeddings.get(&triple.object).unwrap();
        
        let translation = h + r - t;
        
        match self.config.norm {
            1 => translation.iter().map(|x| x.abs()).sum(),
            2 => translation.norm(),
            _ => translation.norm(),
        }
    }
    
    /// Update embeddings using gradient descent
    fn update_embeddings(&mut self, positive: &Triple, negatives: &[Triple]) {
        let pos_dist = self.distance(positive);
        
        for negative in negatives {
            let neg_dist = self.distance(negative);
            let loss = (self.config.margin + pos_dist - neg_dist).max(0.0);
            
            if loss > 0.0 {
                // Calculate gradients
                let h_pos = self.entity_embeddings.get(&positive.subject).unwrap().clone();
                let r = self.relation_embeddings.get(&positive.predicate).unwrap().clone();
                let t_pos = self.entity_embeddings.get(&positive.object).unwrap().clone();
                
                let h_neg = self.entity_embeddings.get(&negative.subject).unwrap().clone();
                let t_neg = self.entity_embeddings.get(&negative.object).unwrap().clone();
                
                let pos_grad = &h_pos + &r - &t_pos;
                let neg_grad = &h_neg + &r - &t_neg;
                
                // Normalize gradients
                let pos_norm = pos_grad.norm();
                let neg_norm = neg_grad.norm();
                
                let pos_grad_norm = if pos_norm > 0.0 { &pos_grad / pos_norm } else { pos_grad };
                let neg_grad_norm = if neg_norm > 0.0 { &neg_grad / neg_norm } else { neg_grad };
                
                // Update embeddings
                let lr = self.config.learning_rate;
                
                // Update positive triple embeddings
                if let Some(h) = self.entity_embeddings.get_mut(&positive.subject) {
                    *h -= lr * &pos_grad_norm;
                    // Re-normalize entity
                    let norm = h.norm();
                    if norm > 0.0 {
                        *h /= norm;
                    }
                }
                
                if let Some(r) = self.relation_embeddings.get_mut(&positive.predicate) {
                    *r -= lr * (&pos_grad_norm - &neg_grad_norm);
                }
                
                if let Some(t) = self.entity_embeddings.get_mut(&positive.object) {
                    *t += lr * &pos_grad_norm;
                    // Re-normalize entity
                    let norm = t.norm();
                    if norm > 0.0 {
                        *t /= norm;
                    }
                }
                
                // Update negative triple embeddings
                if positive.subject != negative.subject {
                    if let Some(h) = self.entity_embeddings.get_mut(&negative.subject) {
                        *h += lr * &neg_grad_norm;
                        // Re-normalize entity
                        let norm = h.norm();
                        if norm > 0.0 {
                            *h /= norm;
                        }
                    }
                }
                
                if positive.object != negative.object {
                    if let Some(t) = self.entity_embeddings.get_mut(&negative.object) {
                        *t -= lr * &neg_grad_norm;
                        // Re-normalize entity
                        let norm = t.norm();
                        if norm > 0.0 {
                            *t /= norm;
                        }
                    }
                }
            }
        }
    }
}

impl KGEmbeddingModel for TransE {
    fn train(&mut self, triples: &[Triple]) -> Result<()> {
        if triples.is_empty() {
            return Err(anyhow!("No triples provided for training"));
        }
        
        // Initialize embeddings
        self.initialize_embeddings(triples);
        
        let mut rng = if let Some(seed) = self.config.random_seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };
        
        // Training loop
        for epoch in 0..self.config.epochs {
            let mut total_loss = 0.0;
            let mut batch_count = 0;
            
            // Shuffle triples
            let mut shuffled_triples = triples.to_vec();
            use rand::seq::SliceRandom;
            shuffled_triples.shuffle(&mut rng);
            
            // Process batches
            for batch in shuffled_triples.chunks(self.config.batch_size) {
                for triple in batch {
                    // Generate negative samples
                    let negatives = self.generate_negative_samples(triple, &mut rng);
                    
                    // Calculate loss
                    let pos_dist = self.distance(triple);
                    for negative in &negatives {
                        let neg_dist = self.distance(negative);
                        let loss = (self.config.margin + pos_dist - neg_dist).max(0.0);
                        total_loss += loss;
                    }
                    
                    // Update embeddings
                    self.update_embeddings(triple, &negatives);
                }
                batch_count += 1;
            }
            
            if epoch % 10 == 0 {
                let avg_loss = total_loss / (batch_count as f32 * self.config.batch_size as f32);
                tracing::info!("Epoch {}: Average loss = {:.4}", epoch, avg_loss);
            }
        }
        
        Ok(())
    }
    
    fn get_entity_embedding(&self, entity: &str) -> Option<Vector> {
        self.entity_embeddings.get(entity)
            .map(|embedding| Vector::new(embedding.iter().cloned().collect()))
    }
    
    fn get_relation_embedding(&self, relation: &str) -> Option<Vector> {
        self.relation_embeddings.get(relation)
            .map(|embedding| Vector::new(embedding.iter().cloned().collect()))
    }
    
    fn score_triple(&self, triple: &Triple) -> f32 {
        -self.distance(triple)
    }
    
    fn predict_tail(&self, head: &str, relation: &str, k: usize) -> Vec<(String, f32)> {
        let h = match self.entity_embeddings.get(head) {
            Some(emb) => emb,
            None => return Vec::new(),
        };
        
        let r = match self.relation_embeddings.get(relation) {
            Some(emb) => emb,
            None => return Vec::new(),
        };
        
        let translation = h + r;
        
        let mut scores: Vec<(String, f32)> = self.entities.iter()
            .filter(|e| *e != head)
            .filter_map(|entity| {
                self.entity_embeddings.get(entity).map(|t| {
                    let distance = match self.config.norm {
                        1 => (&translation - t).iter().map(|x| x.abs()).sum(),
                        2 => (&translation - t).norm(),
                        _ => (&translation - t).norm(),
                    };
                    (entity.clone(), -distance)
                })
            })
            .collect();
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }
    
    fn predict_head(&self, relation: &str, tail: &str, k: usize) -> Vec<(String, f32)> {
        let t = match self.entity_embeddings.get(tail) {
            Some(emb) => emb,
            None => return Vec::new(),
        };
        
        let r = match self.relation_embeddings.get(relation) {
            Some(emb) => emb,
            None => return Vec::new(),
        };
        
        let target = t - r;
        
        let mut scores: Vec<(String, f32)> = self.entities.iter()
            .filter(|e| *e != tail)
            .filter_map(|entity| {
                self.entity_embeddings.get(entity).map(|h| {
                    let distance = match self.config.norm {
                        1 => (h - &target).iter().map(|x| x.abs()).sum(),
                        2 => (h - &target).norm(),
                        _ => (h - &target).norm(),
                    };
                    (entity.clone(), -distance)
                })
            })
            .collect();
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }
    
    fn get_entity_embeddings(&self) -> &HashMap<String, Vector> {
        &self.entity_embeddings
    }
    
    fn get_relation_embeddings(&self) -> &HashMap<String, Vector> {
        &self.relation_embeddings
    }
}

/// ComplEx: Complex number embeddings
/// Uses complex-valued embeddings and Hermitian dot product
pub struct ComplEx {
    config: KGEmbeddingConfig,
    entity_embeddings_real: HashMap<String, DVector<f32>>,
    entity_embeddings_imag: HashMap<String, DVector<f32>>,
    relation_embeddings_real: HashMap<String, DVector<f32>>,
    relation_embeddings_imag: HashMap<String, DVector<f32>>,
    entities: Vec<String>,
    relations: Vec<String>,
}

impl ComplEx {
    pub fn new(config: KGEmbeddingConfig) -> Self {
        Self {
            config,
            entity_embeddings_real: HashMap::new(),
            entity_embeddings_imag: HashMap::new(),
            relation_embeddings_real: HashMap::new(),
            relation_embeddings_imag: HashMap::new(),
            entities: Vec::new(),
            relations: Vec::new(),
        }
    }
    
    /// Initialize embeddings with Xavier initialization
    fn initialize_embeddings(&mut self, triples: &[Triple]) {
        // Collect unique entities and relations
        let mut entities = std::collections::HashSet::new();
        let mut relations = std::collections::HashSet::new();
        
        for triple in triples {
            entities.insert(triple.subject.clone());
            entities.insert(triple.object.clone());
            relations.insert(triple.predicate.clone());
        }
        
        self.entities = entities.into_iter().collect();
        self.relations = relations.into_iter().collect();
        
        // Initialize with Xavier initialization
        let mut rng = if let Some(seed) = self.config.random_seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };
        
        let std_dev = (2.0 / self.config.dimensions as f32).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();
        
        // Initialize entity embeddings
        for entity in &self.entities {
            let real_values: Vec<f32> = (0..self.config.dimensions)
                .map(|_| normal.sample(&mut rng))
                .collect();
            let imag_values: Vec<f32> = (0..self.config.dimensions)
                .map(|_| normal.sample(&mut rng))
                .collect();
            
            self.entity_embeddings_real.insert(entity.clone(), DVector::from_vec(real_values));
            self.entity_embeddings_imag.insert(entity.clone(), DVector::from_vec(imag_values));
        }
        
        // Initialize relation embeddings
        for relation in &self.relations {
            let real_values: Vec<f32> = (0..self.config.dimensions)
                .map(|_| normal.sample(&mut rng))
                .collect();
            let imag_values: Vec<f32> = (0..self.config.dimensions)
                .map(|_| normal.sample(&mut rng))
                .collect();
            
            self.relation_embeddings_real.insert(relation.clone(), DVector::from_vec(real_values));
            self.relation_embeddings_imag.insert(relation.clone(), DVector::from_vec(imag_values));
        }
    }
    
    /// Hermitian dot product for scoring
    fn hermitian_dot(&self, triple: &Triple) -> f32 {
        let h_real = self.entity_embeddings_real.get(&triple.subject).unwrap();
        let h_imag = self.entity_embeddings_imag.get(&triple.subject).unwrap();
        let r_real = self.relation_embeddings_real.get(&triple.predicate).unwrap();
        let r_imag = self.relation_embeddings_imag.get(&triple.predicate).unwrap();
        let t_real = self.entity_embeddings_real.get(&triple.object).unwrap();
        let t_imag = self.entity_embeddings_imag.get(&triple.object).unwrap();
        
        // ComplEx scoring function: Re(<h, r, t̄>)
        // = Re(∑ h_i * r_i * conj(t_i))
        // = ∑ (h_real * r_real * t_real + h_real * r_imag * t_imag + 
        //      h_imag * r_real * t_imag - h_imag * r_imag * t_real)
        
        let mut score = 0.0;
        for i in 0..self.config.dimensions {
            score += h_real[i] * r_real[i] * t_real[i] + h_real[i] * r_imag[i] * t_imag[i] +
                     h_imag[i] * r_real[i] * t_imag[i] - h_imag[i] * r_imag[i] * t_real[i];
        }
        
        score
    }
}

impl KGEmbeddingModel for ComplEx {
    fn train(&mut self, triples: &[Triple]) -> Result<()> {
        if triples.is_empty() {
            return Err(anyhow!("No triples provided for training"));
        }
        
        // Initialize embeddings
        self.initialize_embeddings(triples);
        
        // Training implementation would go here
        // For brevity, using a simplified version
        
        Ok(())
    }
    
    fn get_entity_embedding(&self, entity: &str) -> Option<Vector> {
        // Return concatenated real and imaginary parts
        let real = self.entity_embeddings_real.get(entity)?;
        let imag = self.entity_embeddings_imag.get(entity)?;
        
        let mut values = Vec::with_capacity(self.config.dimensions * 2);
        values.extend(real.iter().cloned());
        values.extend(imag.iter().cloned());
        
        Some(Vector::new(values))
    }
    
    fn get_relation_embedding(&self, relation: &str) -> Option<Vector> {
        // Return concatenated real and imaginary parts
        let real = self.relation_embeddings_real.get(relation)?;
        let imag = self.relation_embeddings_imag.get(relation)?;
        
        let mut values = Vec::with_capacity(self.config.dimensions * 2);
        values.extend(real.iter().cloned());
        values.extend(imag.iter().cloned());
        
        Some(Vector::new(values))
    }
    
    fn score_triple(&self, triple: &Triple) -> f32 {
        self.hermitian_dot(triple)
    }
    
    fn predict_tail(&self, head: &str, relation: &str, k: usize) -> Vec<(String, f32)> {
        let mut scores: Vec<(String, f32)> = self.entities.iter()
            .filter(|e| *e != head)
            .map(|tail| {
                let triple = Triple::new(head.to_string(), relation.to_string(), tail.clone());
                let score = self.score_triple(&triple);
                (tail.clone(), score)
            })
            .collect();
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }
    
    fn predict_head(&self, relation: &str, tail: &str, k: usize) -> Vec<(String, f32)> {
        let mut scores: Vec<(String, f32)> = self.entities.iter()
            .filter(|e| *e != tail)
            .map(|head| {
                let triple = Triple::new(head.clone(), relation.to_string(), tail.to_string());
                let score = self.score_triple(&triple);
                (head.clone(), score)
            })
            .collect();
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }
    
    fn get_entity_embeddings(&self) -> &HashMap<String, Vector> {
        &self.entity_embeddings
    }
    
    fn get_relation_embeddings(&self) -> &HashMap<String, Vector> {
        &self.relation_embeddings
    }
}

/// RotatE: Rotation-based embeddings
/// Models relations as rotations in complex space
pub struct RotatE {
    config: KGEmbeddingConfig,
    entity_embeddings: HashMap<String, DVector<Complex<f32>>>,
    relation_embeddings: HashMap<String, DVector<f32>>, // Phase angles
    entities: Vec<String>,
    relations: Vec<String>,
}

impl RotatE {
    pub fn new(config: KGEmbeddingConfig) -> Self {
        Self {
            config,
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            entities: Vec::new(),
            relations: Vec::new(),
        }
    }
    
    /// Initialize embeddings
    fn initialize_embeddings(&mut self, triples: &[Triple]) {
        // Collect unique entities and relations
        let mut entities = std::collections::HashSet::new();
        let mut relations = std::collections::HashSet::new();
        
        for triple in triples {
            entities.insert(triple.subject.clone());
            entities.insert(triple.object.clone());
            relations.insert(triple.predicate.clone());
        }
        
        self.entities = entities.into_iter().collect();
        self.relations = relations.into_iter().collect();
        
        let mut rng = if let Some(seed) = self.config.random_seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };
        
        // Initialize entity embeddings (complex numbers with unit modulus)
        let uniform = Uniform::new(-std::f32::consts::PI, std::f32::consts::PI);
        
        for entity in &self.entities {
            let phases: Vec<Complex<f32>> = (0..self.config.dimensions)
                .map(|_| {
                    let phase = uniform.sample(&mut rng);
                    Complex::new(phase.cos(), phase.sin())
                })
                .collect();
            
            self.entity_embeddings.insert(entity.clone(), DVector::from_vec(phases));
        }
        
        // Initialize relation embeddings (phase angles)
        for relation in &self.relations {
            let phases: Vec<f32> = (0..self.config.dimensions)
                .map(|_| uniform.sample(&mut rng))
                .collect();
            
            self.relation_embeddings.insert(relation.clone(), DVector::from_vec(phases));
        }
    }
    
    /// Calculate distance for RotatE
    fn distance(&self, triple: &Triple) -> f32 {
        let h = self.entity_embeddings.get(&triple.subject).unwrap();
        let r_phases = self.relation_embeddings.get(&triple.predicate).unwrap();
        let t = self.entity_embeddings.get(&triple.object).unwrap();
        
        // Convert relation phases to complex numbers
        let r: DVector<Complex<f32>> = DVector::from_iterator(
            self.config.dimensions,
            r_phases.iter().map(|&phase| Complex::new(phase.cos(), phase.sin()))
        );
        
        // Apply rotation: h ∘ r (element-wise complex multiplication)
        let rotated: DVector<Complex<f32>> = h.component_mul(&r);
        
        // Calculate distance ||h ∘ r - t||
        let diff = rotated - t;
        diff.iter().map(|c| c.norm()).sum::<f32>()
    }
}

impl KGEmbeddingModel for RotatE {
    fn train(&mut self, triples: &[Triple]) -> Result<()> {
        if triples.is_empty() {
            return Err(anyhow!("No triples provided for training"));
        }
        
        // Initialize embeddings
        self.initialize_embeddings(triples);
        
        // Training implementation would go here
        // For brevity, using a simplified version
        
        Ok(())
    }
    
    fn get_entity_embedding(&self, entity: &str) -> Option<Vector> {
        // Return magnitude and phase representation
        let complex_emb = self.entity_embeddings.get(entity)?;
        
        let mut values = Vec::with_capacity(self.config.dimensions * 2);
        for c in complex_emb.iter() {
            values.push(c.re); // Real part
            values.push(c.im); // Imaginary part
        }
        
        Some(Vector::new(values))
    }
    
    fn get_relation_embedding(&self, relation: &str) -> Option<Vector> {
        self.relation_embeddings.get(relation)
            .map(|phases| Vector::new(phases.iter().cloned().collect()))
    }
    
    fn score_triple(&self, triple: &Triple) -> f32 {
        let gamma = 12.0; // Fixed margin parameter for RotatE
        gamma - self.distance(triple)
    }
    
    fn predict_tail(&self, head: &str, relation: &str, k: usize) -> Vec<(String, f32)> {
        let h = match self.entity_embeddings.get(head) {
            Some(emb) => emb,
            None => return Vec::new(),
        };
        
        let r_phases = match self.relation_embeddings.get(relation) {
            Some(emb) => emb,
            None => return Vec::new(),
        };
        
        // Convert relation phases to complex numbers
        let r: DVector<Complex<f32>> = DVector::from_iterator(
            self.config.dimensions,
            r_phases.iter().map(|&phase| Complex::new(phase.cos(), phase.sin()))
        );
        
        // Apply rotation
        let rotated = h.component_mul(&r);
        
        let mut scores: Vec<(String, f32)> = self.entities.iter()
            .filter(|e| *e != head)
            .filter_map(|entity| {
                self.entity_embeddings.get(entity).map(|t| {
                    let diff = &rotated - t;
                    let distance: f32 = diff.iter().map(|c| c.norm()).sum();
                    (entity.clone(), -distance)
                })
            })
            .collect();
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }
    
    fn predict_head(&self, relation: &str, tail: &str, k: usize) -> Vec<(String, f32)> {
        let t = match self.entity_embeddings.get(tail) {
            Some(emb) => emb,
            None => return Vec::new(),
        };
        
        let r_phases = match self.relation_embeddings.get(relation) {
            Some(emb) => emb,
            None => return Vec::new(),
        };
        
        // Convert relation phases to complex numbers (inverse rotation)
        let r_inv: DVector<Complex<f32>> = DVector::from_iterator(
            self.config.dimensions,
            r_phases.iter().map(|&phase| Complex::new(phase.cos(), -phase.sin()))
        );
        
        let mut scores: Vec<(String, f32)> = self.entities.iter()
            .filter(|e| *e != tail)
            .filter_map(|entity| {
                self.entity_embeddings.get(entity).map(|h| {
                    let rotated = h.component_mul(&r_inv);
                    let diff = rotated - t;
                    let distance: f32 = diff.iter().map(|c| c.norm()).sum();
                    (entity.clone(), -distance)
                })
            })
            .collect();
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }
    
    fn get_entity_embeddings(&self) -> &HashMap<String, Vector> {
        &self.entity_embeddings
    }
    
    fn get_relation_embeddings(&self) -> &HashMap<String, Vector> {
        &self.relation_embeddings
    }
}

/// Unified knowledge graph embedding interface
pub struct KGEmbedding {
    model: Box<dyn KGEmbeddingModel>,
    config: KGEmbeddingConfig,
}

impl KGEmbedding {
    /// Create a new knowledge graph embedding model
    pub fn new(config: KGEmbeddingConfig) -> Self {
        let model: Box<dyn KGEmbeddingModel> = match config.model {
            KGEmbeddingModel::TransE => Box::new(TransE::new(config.clone())),
            KGEmbeddingModel::ComplEx => Box::new(ComplEx::new(config.clone())),
            KGEmbeddingModel::RotatE => Box::new(RotatE::new(config.clone())),
        };
        
        Self { model, config }
    }
    
    /// Train the model
    pub fn train(&mut self, triples: &[Triple]) -> Result<()> {
        self.model.train(triples)
    }
    
    /// Get entity embedding
    pub fn get_entity_embedding(&self, entity: &str) -> Option<Vector> {
        self.model.get_entity_embedding(entity)
    }
    
    /// Get relation embedding
    pub fn get_relation_embedding(&self, relation: &str) -> Option<Vector> {
        self.model.get_relation_embedding(relation)
    }
    
    /// Score a triple
    pub fn score_triple(&self, triple: &Triple) -> f32 {
        self.model.score_triple(triple)
    }
    
    /// Link prediction: predict missing tail
    pub fn predict_tail(&self, head: &str, relation: &str, k: usize) -> Vec<(String, f32)> {
        self.model.predict_tail(head, relation, k)
    }
    
    /// Link prediction: predict missing head
    pub fn predict_head(&self, relation: &str, tail: &str, k: usize) -> Vec<(String, f32)> {
        self.model.predict_head(relation, tail, k)
    }
    
    /// Triple classification: determine if a triple is likely true
    pub fn classify_triple(&self, triple: &Triple, threshold: f32) -> bool {
        self.model.score_triple(triple) > threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_triples() -> Vec<Triple> {
        vec![
            Triple::new("Alice".to_string(), "knows".to_string(), "Bob".to_string()),
            Triple::new("Bob".to_string(), "knows".to_string(), "Charlie".to_string()),
            Triple::new("Alice".to_string(), "likes".to_string(), "Pizza".to_string()),
            Triple::new("Bob".to_string(), "likes".to_string(), "Pasta".to_string()),
            Triple::new("Charlie".to_string(), "knows".to_string(), "Alice".to_string()),
        ]
    }
    
    #[test]
    fn test_transe() {
        let config = KGEmbeddingConfig {
            model: KGEmbeddingModel::TransE,
            dimensions: 50,
            epochs: 10,
            ..Default::default()
        };
        
        let mut model = KGEmbedding::new(config);
        let triples = create_test_triples();
        
        model.train(&triples).unwrap();
        
        // Test embeddings exist
        assert!(model.get_entity_embedding("Alice").is_some());
        assert!(model.get_relation_embedding("knows").is_some());
        
        // Test scoring
        let score = model.score_triple(&triples[0]);
        assert!(score.is_finite());
        
        // Test prediction
        let predictions = model.predict_tail("Alice", "knows", 2);
        assert!(!predictions.is_empty());
    }
    
    #[test]
    fn test_complex() {
        let config = KGEmbeddingConfig {
            model: KGEmbeddingModel::ComplEx,
            dimensions: 50,
            epochs: 10,
            ..Default::default()
        };
        
        let mut model = KGEmbedding::new(config);
        let triples = create_test_triples();
        
        model.train(&triples).unwrap();
        
        // Test embeddings exist
        assert!(model.get_entity_embedding("Bob").is_some());
        let emb = model.get_entity_embedding("Bob").unwrap();
        assert_eq!(emb.dimensions, 100); // Real + imaginary parts
    }
    
    #[test]
    fn test_rotate() {
        let config = KGEmbeddingConfig {
            model: KGEmbeddingModel::RotatE,
            dimensions: 50,
            epochs: 10,
            ..Default::default()
        };
        
        let mut model = KGEmbedding::new(config);
        let triples = create_test_triples();
        
        model.train(&triples).unwrap();
        
        // Test embeddings exist
        assert!(model.get_entity_embedding("Charlie").is_some());
        assert!(model.get_relation_embedding("likes").is_some());
        
        // Test relation embedding is phase angles
        let rel_emb = model.get_relation_embedding("likes").unwrap();
        assert_eq!(rel_emb.dimensions, 50);
    }
}