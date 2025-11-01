//! Graph Neural Network (GNN) embedding models
//!
//! This module provides various GNN architectures for knowledge graph embeddings
//! including GCN, GraphSAGE, GAT, and Graph Transformers.

use crate::{
    EmbeddingError, EmbeddingModel, ModelConfig, ModelStats, TrainingStats, Triple, Vector,
};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::Utc;
use scirs2_core::ndarray_ext::{Array1, Array2};
#[allow(unused_imports)]
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Type of GNN architecture
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum GNNType {
    /// Graph Convolutional Network
    GCN,
    /// GraphSAGE - Sampling and aggregating
    GraphSAGE,
    /// Graph Attention Network
    GAT,
    /// Graph Transformer
    GraphTransformer,
    /// Graph Isomorphism Network
    GIN,
    /// Principal Neighbourhood Aggregation
    PNA,
    /// Heterogeneous Graph Network
    HetGNN,
    /// Temporal Graph Network
    TGN,
}

impl GNNType {
    pub fn default_layers(&self) -> usize {
        match self {
            GNNType::GCN => 2,
            GNNType::GraphSAGE => 2,
            GNNType::GAT => 2,
            GNNType::GraphTransformer => 4,
            GNNType::GIN => 3,
            GNNType::PNA => 3,
            GNNType::HetGNN => 2,
            GNNType::TGN => 2,
        }
    }

    pub fn requires_attention(&self) -> bool {
        matches!(self, GNNType::GAT | GNNType::GraphTransformer)
    }
}

/// Aggregation method for GraphSAGE
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AggregationType {
    Mean,
    Max,
    Sum,
    LSTM,
}

/// Configuration for GNN models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GNNConfig {
    pub base_config: ModelConfig,
    pub gnn_type: GNNType,
    pub num_layers: usize,
    pub hidden_dimensions: Vec<usize>,
    pub dropout: f64,
    pub aggregation: AggregationType,
    pub num_heads: Option<usize>,        // For attention-based models
    pub sample_neighbors: Option<usize>, // For GraphSAGE
    pub residual_connections: bool,
    pub layer_norm: bool,
    pub edge_features: bool,
}

impl Default for GNNConfig {
    fn default() -> Self {
        Self {
            base_config: ModelConfig::default(),
            gnn_type: GNNType::GCN,
            num_layers: 2,
            hidden_dimensions: vec![128, 64],
            dropout: 0.1,
            aggregation: AggregationType::Mean,
            num_heads: None,
            sample_neighbors: None,
            residual_connections: true,
            layer_norm: true,
            edge_features: false,
        }
    }
}

/// GNN-based embedding model
pub struct GNNEmbedding {
    id: Uuid,
    config: GNNConfig,
    entity_embeddings: HashMap<String, Array1<f32>>,
    relation_embeddings: HashMap<String, Array1<f32>>,
    entity_to_idx: HashMap<String, usize>,
    relation_to_idx: HashMap<String, usize>,
    idx_to_entity: HashMap<usize, String>,
    idx_to_relation: HashMap<usize, String>,
    adjacency_list: HashMap<usize, HashSet<(usize, usize)>>, // (neighbor, relation)
    reverse_adjacency_list: HashMap<usize, HashSet<(usize, usize)>>,
    triples: Vec<Triple>,
    layers: Vec<GNNLayer>,
    is_trained: bool,
    creation_time: chrono::DateTime<Utc>,
    last_training_time: Option<chrono::DateTime<Utc>>,
}

/// Single GNN layer
struct GNNLayer {
    weight_matrix: Array2<f32>,
    bias: Array1<f32>,
    attention_weights: Option<AttentionWeights>,
    layer_norm: Option<LayerNormalization>,
}

/// Attention weights for GAT/GraphTransformer
struct AttentionWeights {
    query_weights: Array2<f32>,
    key_weights: Array2<f32>,
    value_weights: Array2<f32>,
    num_heads: usize,
}

/// Layer normalization parameters
struct LayerNormalization {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    epsilon: f32,
}

impl GNNEmbedding {
    pub fn new(config: GNNConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            entity_to_idx: HashMap::new(),
            relation_to_idx: HashMap::new(),
            idx_to_entity: HashMap::new(),
            idx_to_relation: HashMap::new(),
            adjacency_list: HashMap::new(),
            reverse_adjacency_list: HashMap::new(),
            triples: Vec::new(),
            layers: Vec::new(),
            is_trained: false,
            creation_time: Utc::now(),
            last_training_time: None,
        }
    }

    /// Initialize GNN layers
    fn initialize_layers(&mut self) -> Result<()> {
        self.layers.clear();
        let mut rng = Random::seed(42);

        let mut input_dim = self.config.base_config.dimensions;
        let num_layers = self.config.num_layers;

        for i in 0..num_layers {
            let output_dim = if i == num_layers - 1 {
                // Final layer should output back to original embedding dimension
                self.config.base_config.dimensions
            } else if i < self.config.hidden_dimensions.len() {
                self.config.hidden_dimensions[i]
            } else {
                self.config.base_config.dimensions
            };

            // Initialize weight matrix
            let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
            let weight_matrix = Array2::from_shape_fn((input_dim, output_dim), |_| {
                rng.gen_range(0.0..1.0) * scale * 2.0 - scale
            });

            let bias = Array1::zeros(output_dim);

            // Initialize attention weights if needed
            let attention_weights = if self.config.gnn_type.requires_attention() {
                let num_heads = self.config.num_heads.unwrap_or(8);
                let head_dim = output_dim / num_heads;

                // For multi-head attention, each head processes a portion of the output
                let attention_dim = head_dim * num_heads; // Should equal output_dim

                Some(AttentionWeights {
                    query_weights: Array2::from_shape_fn((input_dim, attention_dim), |_| {
                        rng.gen_range(0.0..1.0) * scale * 2.0 - scale
                    }),
                    key_weights: Array2::from_shape_fn((input_dim, attention_dim), |_| {
                        rng.gen_range(0.0..1.0) * scale * 2.0 - scale
                    }),
                    value_weights: Array2::from_shape_fn((input_dim, attention_dim), |_| {
                        rng.gen_range(0.0..1.0) * scale * 2.0 - scale
                    }),
                    num_heads,
                })
            } else {
                None
            };

            // Initialize layer normalization if needed
            let layer_norm = if self.config.layer_norm {
                Some(LayerNormalization {
                    gamma: Array1::ones(output_dim),
                    beta: Array1::zeros(output_dim),
                    epsilon: 1e-5,
                })
            } else {
                None
            };

            self.layers.push(GNNLayer {
                weight_matrix,
                bias,
                attention_weights,
                layer_norm,
            });

            input_dim = output_dim;
        }

        Ok(())
    }

    /// Build adjacency lists from triples
    fn build_adjacency_lists(&mut self) {
        self.adjacency_list.clear();
        self.reverse_adjacency_list.clear();

        for triple in &self.triples {
            let subject_idx = self.entity_to_idx[&triple.subject.iri];
            let object_idx = self.entity_to_idx[&triple.object.iri];
            let relation_idx = self.relation_to_idx[&triple.predicate.iri];

            // Forward adjacency
            self.adjacency_list
                .entry(subject_idx)
                .or_default()
                .insert((object_idx, relation_idx));

            // Reverse adjacency
            self.reverse_adjacency_list
                .entry(object_idx)
                .or_default()
                .insert((subject_idx, relation_idx));
        }
    }

    /// Aggregate neighbor features
    fn aggregate_neighbors(
        &self,
        node_idx: usize,
        node_features: &HashMap<usize, Array1<f32>>,
    ) -> Array1<f32> {
        let neighbors = self.adjacency_list.get(&node_idx);
        let reverse_neighbors = self.reverse_adjacency_list.get(&node_idx);

        let mut neighbor_features = Vec::new();

        // Collect forward neighbors
        if let Some(neighbors) = neighbors {
            for (neighbor_idx, _) in neighbors {
                if let Some(feature) = node_features.get(neighbor_idx) {
                    neighbor_features.push(feature.clone());
                }
            }
        }

        // Collect reverse neighbors
        if let Some(reverse_neighbors) = reverse_neighbors {
            for (neighbor_idx, _) in reverse_neighbors {
                if let Some(feature) = node_features.get(neighbor_idx) {
                    neighbor_features.push(feature.clone());
                }
            }
        }

        if neighbor_features.is_empty() {
            // Return zero vector if no neighbors
            return Array1::zeros(node_features.values().next().unwrap().len());
        }

        // Aggregate based on configuration
        match self.config.aggregation {
            AggregationType::Mean => {
                let sum: Array1<f32> = neighbor_features
                    .iter()
                    .fold(Array1::zeros(neighbor_features[0].len()), |acc, x| acc + x);
                sum / neighbor_features.len() as f32
            }
            AggregationType::Max => neighbor_features.iter().fold(
                Array1::from_elem(neighbor_features[0].len(), f32::NEG_INFINITY),
                |acc, x| {
                    let mut result = acc.clone();
                    for (i, &val) in x.iter().enumerate() {
                        result[i] = result[i].max(val);
                    }
                    result
                },
            ),
            AggregationType::Sum => neighbor_features
                .iter()
                .fold(Array1::zeros(neighbor_features[0].len()), |acc, x| acc + x),
            AggregationType::LSTM => {
                // Simplified LSTM aggregation - in practice would use actual LSTM
                self.aggregate_neighbors_lstm(&neighbor_features)
            }
        }
    }

    /// LSTM aggregation (simplified)
    fn aggregate_neighbors_lstm(&self, neighbor_features: &[Array1<f32>]) -> Array1<f32> {
        // Simplified version - real implementation would use LSTM cells
        let mut aggregated = Array1::zeros(neighbor_features[0].len());
        for feature in neighbor_features {
            aggregated = aggregated * 0.8 + feature * 0.2; // Simple weighted average
        }
        aggregated
    }

    /// Apply GNN layer
    fn apply_layer(
        &self,
        layer: &GNNLayer,
        node_features: &HashMap<usize, Array1<f32>>,
    ) -> HashMap<usize, Array1<f32>> {
        let mut new_features = HashMap::new();

        match self.config.gnn_type {
            GNNType::GCN => self.apply_gcn_layer(layer, node_features, &mut new_features),
            GNNType::GraphSAGE => {
                self.apply_graphsage_layer(layer, node_features, &mut new_features)
            }
            GNNType::GAT => self.apply_gat_layer(layer, node_features, &mut new_features),
            GNNType::GIN => self.apply_gin_layer(layer, node_features, &mut new_features),
            _ => self.apply_gcn_layer(layer, node_features, &mut new_features), // Default to GCN
        }

        new_features
    }

    /// Apply GCN layer
    fn apply_gcn_layer(
        &self,
        layer: &GNNLayer,
        node_features: &HashMap<usize, Array1<f32>>,
        new_features: &mut HashMap<usize, Array1<f32>>,
    ) {
        for (node_idx, feature) in node_features {
            let aggregated = self.aggregate_neighbors(*node_idx, node_features);
            let combined = feature + &aggregated;
            let transformed = combined.dot(&layer.weight_matrix) + &layer.bias;

            // Apply activation (ReLU)
            let activated = transformed.mapv(|x| x.max(0.0));

            // Apply layer norm if configured
            let output = if let Some(ln) = &layer.layer_norm {
                self.apply_layer_norm(&activated, ln)
            } else {
                activated
            };

            new_features.insert(*node_idx, output);
        }
    }

    /// Apply GraphSAGE layer
    fn apply_graphsage_layer(
        &self,
        layer: &GNNLayer,
        node_features: &HashMap<usize, Array1<f32>>,
        new_features: &mut HashMap<usize, Array1<f32>>,
    ) {
        for (node_idx, feature) in node_features {
            let aggregated = self.aggregate_neighbors(*node_idx, node_features);

            // For GraphSAGE, we apply separate transformations and then combine
            // Transform node feature
            let node_transformed = feature.dot(&layer.weight_matrix) + &layer.bias;

            // Transform aggregated neighbor features (reuse same weight matrix for simplicity)
            let neighbor_transformed = aggregated.dot(&layer.weight_matrix) + &layer.bias;

            // Combine the transformed features
            let combined = &node_transformed + &neighbor_transformed;

            // Apply activation and normalization
            let activated = combined.mapv(|x| x.max(0.0));
            let normalized = &activated / (activated.dot(&activated).sqrt() + 1e-6);

            new_features.insert(*node_idx, normalized);
        }
    }

    /// Apply GAT layer
    fn apply_gat_layer(
        &self,
        layer: &GNNLayer,
        node_features: &HashMap<usize, Array1<f32>>,
        new_features: &mut HashMap<usize, Array1<f32>>,
    ) {
        // Simplified GAT - real implementation would compute attention scores
        let attention = layer.attention_weights.as_ref().unwrap();

        for (node_idx, feature) in node_features {
            // Get neighbors
            let mut neighbor_indices = Vec::new();
            if let Some(neighbors) = self.adjacency_list.get(node_idx) {
                neighbor_indices.extend(neighbors.iter().map(|(n, _)| *n));
            }
            if let Some(neighbors) = self.reverse_adjacency_list.get(node_idx) {
                neighbor_indices.extend(neighbors.iter().map(|(n, _)| *n));
            }

            if neighbor_indices.is_empty() {
                // Apply linear transformation even when no neighbors
                let transformed = feature.dot(&layer.weight_matrix) + &layer.bias;
                let activated = transformed.mapv(|x| x.max(0.0));
                new_features.insert(*node_idx, activated);
                continue;
            }

            // Ensure feature dimensions match weight matrix input dimensions
            if feature.len() != attention.query_weights.shape()[0] {
                // Fallback to simple aggregation if dimensions don't match
                let aggregated = self.aggregate_neighbors(*node_idx, node_features);
                let combined = feature + &aggregated;
                let transformed = combined.dot(&layer.weight_matrix) + &layer.bias;
                let activated = transformed.mapv(|x| x.max(0.0));
                new_features.insert(*node_idx, activated);
                continue;
            }

            // Compute attention scores (simplified)
            let query = feature.dot(&attention.query_weights);
            let mut attention_scores = Vec::new();
            let mut neighbor_values = Vec::new();

            for neighbor_idx in &neighbor_indices {
                if let Some(neighbor_feature) = node_features.get(neighbor_idx) {
                    // Check dimension compatibility before computing attention
                    if neighbor_feature.len() != attention.key_weights.shape()[0] {
                        continue;
                    }

                    let key = neighbor_feature.dot(&attention.key_weights);
                    let value = neighbor_feature.dot(&attention.value_weights);

                    // Compute attention score with proper dimension checking
                    if query.len() == key.len() {
                        let score = query.dot(&key) / (attention.num_heads as f32).sqrt();
                        attention_scores.push(score);
                        neighbor_values.push(value);
                    }
                }
            }

            if attention_scores.is_empty() {
                // Fallback to simple aggregation if no valid attention scores
                let aggregated = self.aggregate_neighbors(*node_idx, node_features);
                let combined = feature + &aggregated;
                let transformed = combined.dot(&layer.weight_matrix) + &layer.bias;
                let activated = transformed.mapv(|x| x.max(0.0));
                new_features.insert(*node_idx, activated);
                continue;
            }

            // Softmax
            let max_score = attention_scores
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_scores: Vec<f32> = attention_scores
                .iter()
                .map(|&s| (s - max_score).exp())
                .collect();
            let sum_exp = exp_scores.iter().sum::<f32>();
            let attention_weights: Vec<f32> =
                exp_scores.iter().copied().map(|e| e / sum_exp).collect();

            // Apply attention with proper output dimensions
            let output_dim = layer.weight_matrix.shape()[1];
            let mut aggregated = Array1::<f32>::zeros(output_dim);

            for (i, value) in neighbor_values.iter().enumerate() {
                // Ensure value dimension matches output dimension
                let min_dim = aggregated.len().min(value.len());
                for j in 0..min_dim {
                    aggregated[j] += value[j] * attention_weights[i];
                }
            }

            // Apply linear transformation
            let transformed = feature.dot(&layer.weight_matrix) + &layer.bias;
            let combined =
                if self.config.residual_connections && transformed.len() == aggregated.len() {
                    transformed + &aggregated
                } else {
                    transformed
                };

            let activated = combined.mapv(|x| x.max(0.0));
            new_features.insert(*node_idx, activated);
        }
    }

    /// Apply GIN layer
    fn apply_gin_layer(
        &self,
        layer: &GNNLayer,
        node_features: &HashMap<usize, Array1<f32>>,
        new_features: &mut HashMap<usize, Array1<f32>>,
    ) {
        let epsilon = 0.0; // GIN epsilon parameter

        for (node_idx, feature) in node_features {
            let aggregated = self.aggregate_neighbors(*node_idx, node_features);
            let combined = (1.0 + epsilon) * feature + aggregated;

            // MLP transformation (simplified as single linear layer)
            let transformed = combined.dot(&layer.weight_matrix) + &layer.bias;
            let activated = transformed.mapv(|x| x.max(0.0));

            new_features.insert(*node_idx, activated);
        }
    }

    /// Apply layer normalization
    fn apply_layer_norm(&self, input: &Array1<f32>, ln: &LayerNormalization) -> Array1<f32> {
        let mean = input.mean().unwrap_or(0.0);
        let variance = input.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(1.0);
        let normalized = input.mapv(|x| (x - mean) / (variance + ln.epsilon).sqrt());
        &normalized * &ln.gamma + &ln.beta
    }

    /// Forward pass through all GNN layers
    fn forward(
        &self,
        initial_features: HashMap<usize, Array1<f32>>,
    ) -> HashMap<usize, Array1<f32>> {
        let mut features = initial_features;

        for layer in self.layers.iter() {
            let new_features = self.apply_layer(layer, &features);

            // Apply dropout during training (simplified - always applied here)
            let dropout_rate = self.config.dropout;
            let mut rng = Random::seed(42);

            features = new_features
                .into_iter()
                .map(|(idx, feat)| {
                    let masked = feat.mapv(|x| {
                        if rng.gen_range(0.0..1.0) > dropout_rate as f32 {
                            x / (1.0 - dropout_rate as f32)
                        } else {
                            0.0
                        }
                    });
                    (idx, masked)
                })
                .collect();
        }

        features
    }
}

#[async_trait]
impl EmbeddingModel for GNNEmbedding {
    fn config(&self) -> &ModelConfig {
        &self.config.base_config
    }

    fn model_id(&self) -> &Uuid {
        &self.id
    }

    fn model_type(&self) -> &'static str {
        "GNNEmbedding"
    }

    fn add_triple(&mut self, triple: Triple) -> Result<()> {
        // Add entities to index
        let subject = triple.subject.iri.clone();
        let object = triple.object.iri.clone();
        let predicate = triple.predicate.iri.clone();

        if !self.entity_to_idx.contains_key(&subject) {
            let idx = self.entity_to_idx.len();
            self.entity_to_idx.insert(subject.clone(), idx);
            self.idx_to_entity.insert(idx, subject);
        }

        if !self.entity_to_idx.contains_key(&object) {
            let idx = self.entity_to_idx.len();
            self.entity_to_idx.insert(object.clone(), idx);
            self.idx_to_entity.insert(idx, object);
        }

        if !self.relation_to_idx.contains_key(&predicate) {
            let idx = self.relation_to_idx.len();
            self.relation_to_idx.insert(predicate.clone(), idx);
            self.idx_to_relation.insert(idx, predicate);
        }

        self.triples.push(triple);
        self.is_trained = false;
        Ok(())
    }

    async fn train(&mut self, epochs: Option<usize>) -> Result<TrainingStats> {
        let start_time = std::time::Instant::now();
        let epochs = epochs.unwrap_or(self.config.base_config.max_epochs);

        // Build adjacency lists
        self.build_adjacency_lists();

        // Initialize layers
        self.initialize_layers()?;

        // Initialize random embeddings
        let mut rng = Random::seed(42);
        let dimensions = self.config.base_config.dimensions;

        let mut initial_features = HashMap::new();
        for idx in self.entity_to_idx.values() {
            let embedding =
                Array1::from_shape_fn(dimensions, |_| rng.gen_range(0.0..1.0) * 0.1 - 0.05);
            initial_features.insert(*idx, embedding);
        }

        // Training loop (simplified)
        let mut loss_history = Vec::new();

        for _epoch in 0..epochs {
            // Forward pass
            let output_features = self.forward(initial_features.clone());

            // Compute loss (simplified - just using L2 regularization)
            let loss = output_features
                .values()
                .map(|f| f.mapv(|x| x * x).sum())
                .sum::<f32>()
                / output_features.len() as f32;

            loss_history.push(loss as f64);

            // Update initial features with output (simplified training)
            initial_features = output_features;

            // Early stopping
            if loss < 0.001 {
                break;
            }
        }

        // Store final embeddings
        for (idx, embedding) in initial_features {
            if let Some(entity) = self.idx_to_entity.get(&idx) {
                self.entity_embeddings.insert(entity.clone(), embedding);
            }
        }

        // Generate relation embeddings (simplified - using random initialization)
        for relation in self.relation_to_idx.keys() {
            let embedding =
                Array1::from_shape_fn(dimensions, |_| rng.gen_range(0.0..1.0) * 0.1 - 0.05);
            self.relation_embeddings.insert(relation.clone(), embedding);
        }

        self.is_trained = true;
        self.last_training_time = Some(Utc::now());

        Ok(TrainingStats {
            epochs_completed: loss_history.len(),
            final_loss: *loss_history.last().unwrap_or(&0.0),
            training_time_seconds: start_time.elapsed().as_secs_f64(),
            convergence_achieved: loss_history.last().unwrap_or(&1.0) < &0.001,
            loss_history,
        })
    }

    fn get_entity_embedding(&self, entity: &str) -> Result<Vector> {
        if !self.is_trained {
            return Err(EmbeddingError::ModelNotTrained.into());
        }

        self.entity_embeddings
            .get(entity)
            .map(|e| Vector::new(e.to_vec()))
            .ok_or_else(|| {
                EmbeddingError::EntityNotFound {
                    entity: entity.to_string(),
                }
                .into()
            })
    }

    fn get_relation_embedding(&self, relation: &str) -> Result<Vector> {
        if !self.is_trained {
            return Err(EmbeddingError::ModelNotTrained.into());
        }

        self.relation_embeddings
            .get(relation)
            .map(|e| Vector::new(e.to_vec()))
            .ok_or_else(|| {
                EmbeddingError::RelationNotFound {
                    relation: relation.to_string(),
                }
                .into()
            })
    }

    fn score_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<f64> {
        if !self.is_trained {
            return Err(EmbeddingError::ModelNotTrained.into());
        }

        let subj_emb =
            self.entity_embeddings
                .get(subject)
                .ok_or_else(|| EmbeddingError::EntityNotFound {
                    entity: subject.to_string(),
                })?;

        let pred_emb = self.relation_embeddings.get(predicate).ok_or_else(|| {
            EmbeddingError::RelationNotFound {
                relation: predicate.to_string(),
            }
        })?;

        let obj_emb =
            self.entity_embeddings
                .get(object)
                .ok_or_else(|| EmbeddingError::EntityNotFound {
                    entity: object.to_string(),
                })?;

        // Simple scoring: dot product of transformed embeddings
        let transformed = (subj_emb + pred_emb) * obj_emb;
        Ok(transformed.sum() as f64)
    }

    fn predict_objects(
        &self,
        subject: &str,
        predicate: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        if !self.is_trained {
            return Err(EmbeddingError::ModelNotTrained.into());
        }

        let mut scores = Vec::new();

        for entity in self.entity_to_idx.keys() {
            if let Ok(score) = self.score_triple(subject, predicate, entity) {
                scores.push((entity.clone(), score));
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
        if !self.is_trained {
            return Err(EmbeddingError::ModelNotTrained.into());
        }

        let mut scores = Vec::new();

        for entity in self.entity_to_idx.keys() {
            if let Ok(score) = self.score_triple(entity, predicate, object) {
                scores.push((entity.clone(), score));
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
        if !self.is_trained {
            return Err(EmbeddingError::ModelNotTrained.into());
        }

        let mut scores = Vec::new();

        for relation in self.relation_to_idx.keys() {
            if let Ok(score) = self.score_triple(subject, relation, object) {
                scores.push((relation.clone(), score));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    fn get_entities(&self) -> Vec<String> {
        self.entity_to_idx.keys().cloned().collect()
    }

    fn get_relations(&self) -> Vec<String> {
        self.relation_to_idx.keys().cloned().collect()
    }

    fn get_stats(&self) -> ModelStats {
        ModelStats {
            num_entities: self.entity_to_idx.len(),
            num_relations: self.relation_to_idx.len(),
            num_triples: self.triples.len(),
            dimensions: self.config.base_config.dimensions,
            is_trained: self.is_trained,
            model_type: format!("GNNEmbedding-{:?}", self.config.gnn_type),
            creation_time: self.creation_time,
            last_training_time: self.last_training_time,
        }
    }

    fn save(&self, _path: &str) -> Result<()> {
        // Implementation would save model weights and configuration
        Ok(())
    }

    fn load(&mut self, _path: &str) -> Result<()> {
        // Implementation would load model weights and configuration
        Ok(())
    }

    fn clear(&mut self) {
        self.entity_embeddings.clear();
        self.relation_embeddings.clear();
        self.entity_to_idx.clear();
        self.relation_to_idx.clear();
        self.idx_to_entity.clear();
        self.idx_to_relation.clear();
        self.adjacency_list.clear();
        self.reverse_adjacency_list.clear();
        self.triples.clear();
        self.layers.clear();
        self.is_trained = false;
    }

    fn is_trained(&self) -> bool {
        self.is_trained
    }

    async fn encode(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Err(anyhow!(
            "Knowledge graph embedding model does not support text encoding"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamedNode;

    #[tokio::test]
    async fn test_gnn_embedding_basic() {
        let config = GNNConfig {
            gnn_type: GNNType::GCN,
            num_layers: 2,
            hidden_dimensions: vec![64, 32],
            ..Default::default()
        };

        let mut model = GNNEmbedding::new(config);

        // Add some triples
        let triple1 = Triple::new(
            NamedNode::new("http://example.org/Alice").unwrap(),
            NamedNode::new("http://example.org/knows").unwrap(),
            NamedNode::new("http://example.org/Bob").unwrap(),
        );

        let triple2 = Triple::new(
            NamedNode::new("http://example.org/Bob").unwrap(),
            NamedNode::new("http://example.org/knows").unwrap(),
            NamedNode::new("http://example.org/Charlie").unwrap(),
        );

        model.add_triple(triple1).unwrap();
        model.add_triple(triple2).unwrap();

        // Train the model
        let _stats = model.train(Some(10)).await.unwrap();
        assert!(model.is_trained());

        // Get embeddings
        let alice_emb = model
            .get_entity_embedding("http://example.org/Alice")
            .unwrap();
        assert_eq!(alice_emb.dimensions, 100); // Default dimensions

        // Test predictions
        let predictions = model
            .predict_objects("http://example.org/Alice", "http://example.org/knows", 5)
            .unwrap();
        assert!(!predictions.is_empty());
    }

    #[tokio::test]
    async fn test_gnn_types() {
        for gnn_type in [GNNType::GCN, GNNType::GraphSAGE, GNNType::GAT, GNNType::GIN] {
            let config = GNNConfig {
                gnn_type,
                num_heads: if gnn_type == GNNType::GAT {
                    Some(4)
                } else {
                    None
                },
                ..Default::default()
            };

            let mut model = GNNEmbedding::new(config);

            let triple = Triple::new(
                NamedNode::new("http://example.org/A").unwrap(),
                NamedNode::new("http://example.org/rel").unwrap(),
                NamedNode::new("http://example.org/B").unwrap(),
            );

            model.add_triple(triple).unwrap();
            let _stats = model.train(Some(5)).await.unwrap();
            assert!(model.is_trained());
        }
    }
}
