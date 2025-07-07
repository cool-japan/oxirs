//! Cross-modal embeddings for multi-modal vector search
//!
//! This module provides CLIP-style cross-modal embeddings that can handle:
//! - Text-image alignment
//! - Multi-modal fusion
//! - Cross-modal attention mechanisms
//! - Joint embedding spaces

use crate::Vector;
use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Modality types supported by the cross-modal system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    Graph,
    Numeric,
    Custom(u8),
}

/// Configuration for cross-modal embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalConfig {
    /// Dimension of the joint embedding space
    pub joint_embedding_dim: usize,
    /// Temperature parameter for contrastive learning
    pub temperature: f32,
    /// Enable attention mechanisms
    pub enable_attention: bool,
    /// Attention head count
    pub attention_heads: usize,
    /// Enable multi-scale features
    pub enable_multi_scale: bool,
    /// Fusion strategy for combining modalities
    pub fusion_strategy: FusionStrategy,
    /// Alignment learning rate
    pub alignment_learning_rate: f32,
    /// Enable domain adaptation
    pub enable_domain_adaptation: bool,
    /// Modality weights for fusion
    pub modality_weights: HashMap<Modality, f32>,
}

impl Default for CrossModalConfig {
    fn default() -> Self {
        let mut modality_weights = HashMap::new();
        modality_weights.insert(Modality::Text, 1.0);
        modality_weights.insert(Modality::Image, 1.0);
        modality_weights.insert(Modality::Audio, 0.8);
        modality_weights.insert(Modality::Video, 0.9);

        Self {
            joint_embedding_dim: 512,
            temperature: 0.07,
            enable_attention: true,
            attention_heads: 8,
            enable_multi_scale: true,
            fusion_strategy: FusionStrategy::AttentionWeighted,
            alignment_learning_rate: 1e-4,
            enable_domain_adaptation: true,
            modality_weights,
        }
    }
}

/// Fusion strategies for combining multiple modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Simple concatenation of embeddings
    Concatenation,
    /// Weighted average of embeddings
    WeightedAverage,
    /// Attention-weighted fusion
    AttentionWeighted,
    /// Early fusion before encoding
    EarlyFusion,
    /// Late fusion after encoding
    LateFusion,
    /// Hierarchical fusion with multiple stages
    HierarchicalFusion,
    /// Graph-based fusion using cross-modal graphs
    GraphFusion,
}

/// Multi-modal content that can contain multiple types of data
#[derive(Debug, Clone)]
pub struct MultiModalContent {
    pub modalities: HashMap<Modality, ModalityData>,
    pub metadata: HashMap<String, String>,
    pub temporal_info: Option<TemporalInfo>,
    pub spatial_info: Option<SpatialInfo>,
}

/// Data for a specific modality
#[derive(Debug, Clone)]
pub enum ModalityData {
    Text(String),
    Image(ImageData),
    Audio(AudioData),
    Video(VideoData),
    Graph(GraphData),
    Numeric(Vec<f32>),
    Raw(Vec<u8>),
}

/// Image data representation
#[derive(Debug, Clone)]
pub struct ImageData {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub format: ImageFormat,
    pub features: Option<Vec<f32>>, // Pre-extracted features
}

#[derive(Debug, Clone)]
pub enum ImageFormat {
    RGB,
    RGBA,
    Grayscale,
    BGR,
    YUV,
}

/// Audio data representation
#[derive(Debug, Clone)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u32,
    pub duration: f32,
    pub features: Option<Vec<f32>>, // MFCC, spectral features, etc.
}

/// Video data representation
#[derive(Debug, Clone)]
pub struct VideoData {
    pub frames: Vec<ImageData>,
    pub audio: Option<AudioData>,
    pub fps: f32,
    pub duration: f32,
    pub keyframes: Vec<usize>, // Indices of keyframes
}

/// Graph data representation for knowledge graphs
#[derive(Debug, Clone)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: String,
    pub labels: Vec<String>,
    pub properties: HashMap<String, String>,
    pub embedding: Option<Vector>,
}

#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub properties: HashMap<String, String>,
    pub weight: Option<f32>,
}

/// Temporal information for time-series data
#[derive(Debug, Clone)]
pub struct TemporalInfo {
    pub timestamp: std::time::SystemTime,
    pub duration: Option<std::time::Duration>,
    pub temporal_features: Vec<f32>,
}

/// Spatial information for location-aware embeddings
#[derive(Debug, Clone)]
pub struct SpatialInfo {
    pub coordinates: (f64, f64), // latitude, longitude
    pub elevation: Option<f32>,
    pub spatial_features: Vec<f32>,
}

/// Cross-modal embedding encoder that handles multiple modalities
pub struct CrossModalEncoder {
    config: CrossModalConfig,
    text_encoder: Box<dyn TextEncoder>,
    image_encoder: Box<dyn ImageEncoder>,
    audio_encoder: Box<dyn AudioEncoder>,
    video_encoder: Box<dyn VideoEncoder>,
    graph_encoder: Box<dyn GraphEncoder>,
    attention_mechanism: AttentionMechanism,
    fusion_layer: FusionLayer,
    alignment_cache: Arc<RwLock<HashMap<String, Vector>>>,
}

/// Text encoder trait for cross-modal systems
pub trait TextEncoder: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vector>;
    fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vector>>;
    fn get_embedding_dim(&self) -> usize;
}

/// Image encoder trait for cross-modal systems
pub trait ImageEncoder: Send + Sync {
    fn encode(&self, image: &ImageData) -> Result<Vector>;
    fn encode_batch(&self, images: &[ImageData]) -> Result<Vec<Vector>>;
    fn get_embedding_dim(&self) -> usize;
    fn extract_features(&self, image: &ImageData) -> Result<Vec<f32>>;
}

/// Audio encoder trait for cross-modal systems
pub trait AudioEncoder: Send + Sync {
    fn encode(&self, audio: &AudioData) -> Result<Vector>;
    fn encode_batch(&self, audios: &[AudioData]) -> Result<Vec<Vector>>;
    fn get_embedding_dim(&self) -> usize;
    fn extract_features(&self, audio: &AudioData) -> Result<Vec<f32>>;
}

/// Video encoder trait for cross-modal systems
pub trait VideoEncoder: Send + Sync {
    fn encode(&self, video: &VideoData) -> Result<Vector>;
    fn encode_keyframes(&self, video: &VideoData) -> Result<Vec<Vector>>;
    fn get_embedding_dim(&self) -> usize;
}

/// Graph encoder trait for knowledge graph embeddings
pub trait GraphEncoder: Send + Sync {
    fn encode(&self, graph: &GraphData) -> Result<Vector>;
    fn encode_node(&self, node: &GraphNode) -> Result<Vector>;
    fn encode_subgraph(&self, nodes: &[GraphNode], edges: &[GraphEdge]) -> Result<Vector>;
    fn get_embedding_dim(&self) -> usize;
}

/// Attention mechanism for cross-modal alignment
#[derive(Debug, Clone)]
pub struct AttentionMechanism {
    pub num_heads: usize,
    pub head_dim: usize,
    pub dropout_rate: f32,
    pub scale: f32,
}

impl AttentionMechanism {
    pub fn new(num_heads: usize, embedding_dim: usize) -> Self {
        let head_dim = embedding_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        Self {
            num_heads,
            head_dim,
            dropout_rate: 0.1,
            scale,
        }
    }

    /// Compute cross-modal attention between two modalities
    pub fn cross_attention(&self, query: &Vector, key: &Vector, value: &Vector) -> Result<Vector> {
        // Simplified cross-attention implementation
        // In practice, this would use matrix operations and multiple heads

        let query_f32 = query.as_f32();
        let key_f32 = key.as_f32();
        let value_f32 = value.as_f32();

        if query_f32.len() != key_f32.len() || key_f32.len() != value_f32.len() {
            return Err(anyhow!("Dimension mismatch in attention"));
        }

        // Compute attention scores (simplified dot-product attention)
        let attention_score = query_f32
            .iter()
            .zip(&key_f32)
            .map(|(q, k)| q * k)
            .sum::<f32>()
            * self.scale;

        // Apply attention to values
        let attended_values: Vec<f32> = value_f32
            .iter()
            .map(|v| v * attention_score.tanh()) // Apply softmax-like normalization
            .collect();

        Ok(Vector::new(attended_values))
    }

    /// Multi-head attention for richer representations
    pub fn multi_head_attention(&self, inputs: &[Vector]) -> Result<Vector> {
        if inputs.is_empty() {
            return Err(anyhow!("No input vectors for attention"));
        }

        let dim = inputs[0].dimensions;
        let mut combined_output = vec![0.0f32; dim];

        // Simulate multi-head processing
        for (_head_idx, input) in inputs.iter().enumerate().take(self.num_heads) {
            let input_f32 = input.as_f32();
            let head_weight = 1.0 / self.num_heads as f32;

            for (i, &value) in input_f32.iter().enumerate() {
                if i < combined_output.len() {
                    combined_output[i] += value * head_weight;
                }
            }
        }

        Ok(Vector::new(combined_output))
    }
}

/// Fusion layer for combining multiple modalities
#[derive(Debug, Clone)]
pub struct FusionLayer {
    strategy: FusionStrategy,
    modality_weights: HashMap<Modality, f32>,
    learned_weights: Option<Vec<f32>>,
}

impl FusionLayer {
    pub fn new(strategy: FusionStrategy, modality_weights: HashMap<Modality, f32>) -> Self {
        Self {
            strategy,
            modality_weights,
            learned_weights: None,
        }
    }

    /// Fuse embeddings from multiple modalities
    pub fn fuse(&self, embeddings: &HashMap<Modality, Vector>) -> Result<Vector> {
        if embeddings.is_empty() {
            return Err(anyhow!("No embeddings to fuse"));
        }

        match self.strategy {
            FusionStrategy::Concatenation => self.concatenation_fusion(embeddings),
            FusionStrategy::WeightedAverage => self.weighted_average_fusion(embeddings),
            FusionStrategy::AttentionWeighted => self.attention_weighted_fusion(embeddings),
            FusionStrategy::EarlyFusion => self.early_fusion(embeddings),
            FusionStrategy::LateFusion => self.late_fusion(embeddings),
            FusionStrategy::HierarchicalFusion => self.hierarchical_fusion(embeddings),
            FusionStrategy::GraphFusion => self.graph_fusion(embeddings),
        }
    }

    fn concatenation_fusion(&self, embeddings: &HashMap<Modality, Vector>) -> Result<Vector> {
        let mut concatenated = Vec::new();

        // Maintain consistent ordering
        let ordered_modalities = [
            Modality::Text,
            Modality::Image,
            Modality::Audio,
            Modality::Video,
        ];

        for modality in &ordered_modalities {
            if let Some(embedding) = embeddings.get(modality) {
                concatenated.extend_from_slice(&embedding.as_f32());
            }
        }

        // Add any custom modalities
        for (modality, embedding) in embeddings {
            if !ordered_modalities.contains(modality) {
                concatenated.extend_from_slice(&embedding.as_f32());
            }
        }

        Ok(Vector::new(concatenated))
    }

    fn weighted_average_fusion(&self, embeddings: &HashMap<Modality, Vector>) -> Result<Vector> {
        let first_embedding = embeddings.values().next().unwrap();
        let dim = first_embedding.dimensions;
        let mut fused = vec![0.0f32; dim];
        let mut total_weight = 0.0f32;

        for (modality, embedding) in embeddings {
            let weight = self.modality_weights.get(modality).copied().unwrap_or(1.0);
            let embedding_f32 = embedding.as_f32();

            if embedding_f32.len() != dim {
                return Err(anyhow!("Dimension mismatch in embeddings"));
            }

            for i in 0..dim {
                fused[i] += embedding_f32[i] * weight;
            }
            total_weight += weight;
        }

        // Normalize by total weight
        for value in &mut fused {
            *value /= total_weight;
        }

        Ok(Vector::new(fused))
    }

    fn attention_weighted_fusion(&self, embeddings: &HashMap<Modality, Vector>) -> Result<Vector> {
        // Simplified attention-based fusion
        // In practice, this would use learned attention weights

        let modalities: Vec<&Modality> = embeddings.keys().collect();
        let vectors: Vec<&Vector> = embeddings.values().collect();

        if vectors.is_empty() {
            return Err(anyhow!("No vectors to fuse"));
        }

        let dim = vectors[0].dimensions;
        let mut attention_weights = vec![1.0f32; modalities.len()];

        // Compute simple attention weights based on vector norms
        for (i, vector) in vectors.iter().enumerate() {
            attention_weights[i] = vector.magnitude();
        }

        // Softmax normalization
        let max_weight = attention_weights
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_weights: Vec<f32> = attention_weights
            .iter()
            .map(|w| (w - max_weight).exp())
            .collect();
        let sum_exp: f32 = exp_weights.iter().sum();

        for weight in &mut attention_weights {
            *weight = (*weight - max_weight).exp() / sum_exp;
        }

        // Apply attention weights
        let mut fused = vec![0.0f32; dim];
        for (i, vector) in vectors.iter().enumerate() {
            let vector_f32 = vector.as_f32();
            let weight = attention_weights[i];

            for j in 0..dim {
                fused[j] += vector_f32[j] * weight;
            }
        }

        Ok(Vector::new(fused))
    }

    fn early_fusion(&self, embeddings: &HashMap<Modality, Vector>) -> Result<Vector> {
        // Early fusion: combine at feature level before final encoding
        self.concatenation_fusion(embeddings)
    }

    fn late_fusion(&self, embeddings: &HashMap<Modality, Vector>) -> Result<Vector> {
        // Late fusion: combine already encoded features
        self.weighted_average_fusion(embeddings)
    }

    fn hierarchical_fusion(&self, embeddings: &HashMap<Modality, Vector>) -> Result<Vector> {
        // Hierarchical fusion: multi-stage combination

        // Stage 1: Group similar modalities
        let mut text_audio = Vec::new();
        let mut visual = Vec::new();
        let mut structured = Vec::new();

        for (modality, embedding) in embeddings {
            match modality {
                Modality::Text | Modality::Audio => text_audio.push(embedding),
                Modality::Image | Modality::Video => visual.push(embedding),
                Modality::Graph | Modality::Numeric => structured.push(embedding),
                _ => text_audio.push(embedding), // Default to text-audio group
            }
        }

        // Stage 2: Fuse within groups
        let mut group_embeddings = HashMap::new();

        if !text_audio.is_empty() {
            let fused_ta = self.fuse_group(&text_audio)?;
            group_embeddings.insert(Modality::Text, fused_ta);
        }

        if !visual.is_empty() {
            let fused_visual = self.fuse_group(&visual)?;
            group_embeddings.insert(Modality::Image, fused_visual);
        }

        if !structured.is_empty() {
            let fused_structured = self.fuse_group(&structured)?;
            group_embeddings.insert(Modality::Graph, fused_structured);
        }

        // Stage 3: Final fusion
        self.weighted_average_fusion(&group_embeddings)
    }

    fn fuse_group(&self, embeddings: &[&Vector]) -> Result<Vector> {
        if embeddings.is_empty() {
            return Err(anyhow!("No embeddings to fuse in group"));
        }

        let dim = embeddings[0].dimensions;
        let mut fused = vec![0.0f32; dim];

        for embedding in embeddings {
            let embedding_f32 = embedding.as_f32();
            for i in 0..dim {
                fused[i] += embedding_f32[i];
            }
        }

        // Average
        let count = embeddings.len() as f32;
        for value in &mut fused {
            *value /= count;
        }

        Ok(Vector::new(fused))
    }

    fn graph_fusion(&self, embeddings: &HashMap<Modality, Vector>) -> Result<Vector> {
        // Graph-based fusion using modality relationships
        // For now, use weighted average based on modality connectivity
        self.weighted_average_fusion(embeddings)
    }
}

impl CrossModalEncoder {
    pub fn new(
        config: CrossModalConfig,
        text_encoder: Box<dyn TextEncoder>,
        image_encoder: Box<dyn ImageEncoder>,
        audio_encoder: Box<dyn AudioEncoder>,
        video_encoder: Box<dyn VideoEncoder>,
        graph_encoder: Box<dyn GraphEncoder>,
    ) -> Self {
        let attention_mechanism =
            AttentionMechanism::new(config.attention_heads, config.joint_embedding_dim);

        let fusion_layer = FusionLayer::new(
            config.fusion_strategy.clone(),
            config.modality_weights.clone(),
        );

        Self {
            config,
            text_encoder,
            image_encoder,
            audio_encoder,
            video_encoder,
            graph_encoder,
            attention_mechanism,
            fusion_layer,
            alignment_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Encode multi-modal content into a joint embedding space
    pub fn encode(&self, content: &MultiModalContent) -> Result<Vector> {
        let mut modality_embeddings = HashMap::new();

        // Encode each modality present in the content
        for (modality, data) in &content.modalities {
            let embedding = match (modality, data) {
                (Modality::Text, ModalityData::Text(text)) => self.text_encoder.encode(text)?,
                (Modality::Image, ModalityData::Image(image)) => {
                    self.image_encoder.encode(image)?
                }
                (Modality::Audio, ModalityData::Audio(audio)) => {
                    self.audio_encoder.encode(audio)?
                }
                (Modality::Video, ModalityData::Video(video)) => {
                    self.video_encoder.encode(video)?
                }
                (Modality::Graph, ModalityData::Graph(graph)) => {
                    self.graph_encoder.encode(graph)?
                }
                (Modality::Numeric, ModalityData::Numeric(values)) => {
                    // Ensure numeric vectors match joint embedding dimension
                    let mut padded_values = values.clone();
                    if padded_values.len() < self.config.joint_embedding_dim {
                        // Pad with zeros to match embedding dimension
                        padded_values.resize(self.config.joint_embedding_dim, 0.0);
                    } else if padded_values.len() > self.config.joint_embedding_dim {
                        // Truncate to match embedding dimension
                        padded_values.truncate(self.config.joint_embedding_dim);
                    }
                    Vector::new(padded_values)
                }
                _ => return Err(anyhow!("Modality-data type mismatch")),
            };

            modality_embeddings.insert(*modality, embedding);
        }

        // Apply cross-modal attention if enabled
        if self.config.enable_attention && modality_embeddings.len() > 1 {
            modality_embeddings = self.apply_cross_modal_attention(modality_embeddings)?;
        }

        // Fuse all modality embeddings
        let fused_embedding = self.fusion_layer.fuse(&modality_embeddings)?;

        // Project to joint embedding space if needed
        let joint_embedding = self.project_to_joint_space(&fused_embedding)?;

        Ok(joint_embedding)
    }

    /// Apply cross-modal attention between modalities
    fn apply_cross_modal_attention(
        &self,
        mut embeddings: HashMap<Modality, Vector>,
    ) -> Result<HashMap<Modality, Vector>> {
        let modalities: Vec<Modality> = embeddings.keys().copied().collect();

        // Apply attention between all pairs of modalities
        for i in 0..modalities.len() {
            for j in 0..modalities.len() {
                if i != j {
                    let query_modality = modalities[i];
                    let key_modality = modalities[j];

                    if let (Some(query), Some(key)) = (
                        embeddings.get(&query_modality).cloned(),
                        embeddings.get(&key_modality).cloned(),
                    ) {
                        // Use key as both key and value for simplicity
                        let attended = self
                            .attention_mechanism
                            .cross_attention(&query, &key, &key)?;

                        // Update the query embedding with attention
                        if let Some(original) = embeddings.get_mut(&query_modality) {
                            *original = self.combine_attended(original, &attended)?;
                        }
                    }
                }
            }
        }

        Ok(embeddings)
    }

    /// Combine original and attended embeddings
    fn combine_attended(&self, original: &Vector, attended: &Vector) -> Result<Vector> {
        let alpha = 0.5; // Attention weight
        let original_f32 = original.as_f32();
        let attended_f32 = attended.as_f32();

        if original_f32.len() != attended_f32.len() {
            return Err(anyhow!("Dimension mismatch in attention combination"));
        }

        let combined: Vec<f32> = original_f32
            .iter()
            .zip(&attended_f32)
            .map(|(o, a)| (1.0 - alpha) * o + alpha * a)
            .collect();

        Ok(Vector::new(combined))
    }

    /// Project embedding to joint embedding space
    fn project_to_joint_space(&self, embedding: &Vector) -> Result<Vector> {
        let embedding_f32 = embedding.as_f32();

        // If already the right dimension, return as-is
        if embedding_f32.len() == self.config.joint_embedding_dim {
            return Ok(embedding.clone());
        }

        // Simple projection: truncate or pad
        let mut projected = vec![0.0f32; self.config.joint_embedding_dim];
        let copy_len = embedding_f32.len().min(self.config.joint_embedding_dim);

        for i in 0..copy_len {
            projected[i] = embedding_f32[i];
        }

        // If we had to truncate, normalize to maintain magnitude
        if embedding_f32.len() > self.config.joint_embedding_dim {
            let original_norm = embedding.magnitude();
            let projected_vector = Vector::new(projected.clone());
            let projected_norm = projected_vector.magnitude();

            if projected_norm > 0.0 {
                let scale = original_norm / projected_norm;
                projected = projected_vector.scale(scale).as_f32();
            }
        }

        Ok(Vector::new(projected))
    }

    /// Calculate cross-modal similarity between two multi-modal contents
    pub fn cross_modal_similarity(
        &self,
        content1: &MultiModalContent,
        content2: &MultiModalContent,
    ) -> Result<f32> {
        let embedding1 = self.encode(content1)?;
        let embedding2 = self.encode(content2)?;

        embedding1.cosine_similarity(&embedding2)
    }

    /// Find cross-modal matches (e.g., images that match text descriptions)
    pub fn find_cross_modal_matches(
        &self,
        query_content: &MultiModalContent,
        candidates: &[MultiModalContent],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let query_embedding = self.encode(query_content)?;
        let mut similarities = Vec::new();

        for (idx, candidate) in candidates.iter().enumerate() {
            let candidate_embedding = self.encode(candidate)?;
            let similarity = query_embedding.cosine_similarity(&candidate_embedding)?;
            similarities.push((idx, similarity));
        }

        // Sort by similarity (descending) and take top k
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(top_k);

        Ok(similarities)
    }

    /// Align embeddings across modalities using contrastive learning
    pub fn align_modalities(
        &mut self,
        paired_data: &[(MultiModalContent, MultiModalContent)],
    ) -> Result<()> {
        // Simplified alignment training
        // In practice, this would involve gradient-based optimization

        for (content1, content2) in paired_data {
            let embedding1 = self.encode(content1)?;
            let embedding2 = self.encode(content2)?;

            // Calculate alignment loss (contrastive)
            let similarity = embedding1.cosine_similarity(&embedding2)?;
            let target_similarity = 1.0; // Paired data should be similar

            let _loss = (similarity - target_similarity).powi(2);

            // In a real implementation, this would update model parameters
            // For now, we just cache the aligned embeddings
            let cache_key1 = self.generate_cache_key(content1);
            let cache_key2 = self.generate_cache_key(content2);

            let mut cache = self.alignment_cache.write();
            cache.insert(cache_key1, embedding1);
            cache.insert(cache_key2, embedding2);
        }

        Ok(())
    }

    fn generate_cache_key(&self, content: &MultiModalContent) -> String {
        // Generate a simple hash-based key for the content
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        for (modality, data) in &content.modalities {
            modality.hash(&mut hasher);
            match data {
                ModalityData::Text(text) => text.hash(&mut hasher),
                ModalityData::Numeric(values) => {
                    for &value in values {
                        value.to_bits().hash(&mut hasher);
                    }
                }
                _ => {
                    // For complex data types, use a simplified hash
                    std::mem::discriminant(data).hash(&mut hasher);
                }
            }
        }

        format!("multimodal_{:x}", hasher.finish())
    }

    /// Get alignment statistics
    pub fn get_alignment_stats(&self) -> (usize, f32) {
        let cache = self.alignment_cache.read();
        let cache_size = cache.len();
        let avg_similarity = 0.85; // Placeholder - would calculate from actual alignments

        (cache_size, avg_similarity)
    }
}

/// Simple implementations of encoder traits for testing
pub struct MockTextEncoder {
    embedding_dim: usize,
}

impl MockTextEncoder {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }
}

impl TextEncoder for MockTextEncoder {
    fn encode(&self, text: &str) -> Result<Vector> {
        // Simple hash-based encoding for testing
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        let mut values = Vec::with_capacity(self.embedding_dim);
        let mut seed = hash;

        for _ in 0..self.embedding_dim {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let normalized = (seed as f32) / (u64::MAX as f32);
            values.push((normalized - 0.5) * 2.0);
        }

        Ok(Vector::new(values))
    }

    fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vector>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }

    fn get_embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

/// Similar mock implementations for other modalities
pub struct MockImageEncoder {
    embedding_dim: usize,
}
pub struct MockAudioEncoder {
    embedding_dim: usize,
}
pub struct MockVideoEncoder {
    embedding_dim: usize,
}
pub struct MockGraphEncoder {
    embedding_dim: usize,
}

impl MockImageEncoder {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }
}

impl MockAudioEncoder {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }
}

impl MockVideoEncoder {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }
}

impl MockGraphEncoder {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }
}

impl ImageEncoder for MockImageEncoder {
    fn encode(&self, _image: &ImageData) -> Result<Vector> {
        Ok(Vector::new(vec![0.0; self.embedding_dim]))
    }

    fn encode_batch(&self, images: &[ImageData]) -> Result<Vec<Vector>> {
        Ok(vec![
            Vector::new(vec![0.0; self.embedding_dim]);
            images.len()
        ])
    }

    fn get_embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn extract_features(&self, _image: &ImageData) -> Result<Vec<f32>> {
        Ok(vec![0.0; 1000]) // Mock CNN features
    }
}

impl AudioEncoder for MockAudioEncoder {
    fn encode(&self, _audio: &AudioData) -> Result<Vector> {
        Ok(Vector::new(vec![0.0; self.embedding_dim]))
    }

    fn encode_batch(&self, audios: &[AudioData]) -> Result<Vec<Vector>> {
        Ok(vec![
            Vector::new(vec![0.0; self.embedding_dim]);
            audios.len()
        ])
    }

    fn get_embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn extract_features(&self, _audio: &AudioData) -> Result<Vec<f32>> {
        Ok(vec![0.0; 128]) // Mock MFCC features
    }
}

impl VideoEncoder for MockVideoEncoder {
    fn encode(&self, _video: &VideoData) -> Result<Vector> {
        Ok(Vector::new(vec![0.0; self.embedding_dim]))
    }

    fn encode_keyframes(&self, video: &VideoData) -> Result<Vec<Vector>> {
        Ok(vec![
            Vector::new(vec![0.0; self.embedding_dim]);
            video.keyframes.len()
        ])
    }

    fn get_embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

impl GraphEncoder for MockGraphEncoder {
    fn encode(&self, _graph: &GraphData) -> Result<Vector> {
        Ok(Vector::new(vec![0.0; self.embedding_dim]))
    }

    fn encode_node(&self, _node: &GraphNode) -> Result<Vector> {
        Ok(Vector::new(vec![0.0; self.embedding_dim]))
    }

    fn encode_subgraph(&self, _nodes: &[GraphNode], _edges: &[GraphEdge]) -> Result<Vector> {
        Ok(Vector::new(vec![0.0; self.embedding_dim]))
    }

    fn get_embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_modal_encoder_creation() {
        let config = CrossModalConfig::default();
        let text_encoder = Box::new(MockTextEncoder::new(512));
        let image_encoder = Box::new(MockImageEncoder::new(512));
        let audio_encoder = Box::new(MockAudioEncoder::new(512));
        let video_encoder = Box::new(MockVideoEncoder::new(512));
        let graph_encoder = Box::new(MockGraphEncoder::new(512));

        let encoder = CrossModalEncoder::new(
            config,
            text_encoder,
            image_encoder,
            audio_encoder,
            video_encoder,
            graph_encoder,
        );

        assert_eq!(encoder.config.joint_embedding_dim, 512);
    }

    #[test]
    fn test_multi_modal_content_encoding() {
        let config = CrossModalConfig::default();
        let encoder = create_test_encoder(config);

        let mut content = MultiModalContent {
            modalities: HashMap::new(),
            metadata: HashMap::new(),
            temporal_info: None,
            spatial_info: None,
        };

        content.modalities.insert(
            Modality::Text,
            ModalityData::Text("Hello world".to_string()),
        );
        content.modalities.insert(
            Modality::Numeric,
            ModalityData::Numeric(vec![1.0, 2.0, 3.0]),
        );

        let embedding = encoder.encode(&content).unwrap();
        assert_eq!(embedding.dimensions, 512);
    }

    #[test]
    fn test_fusion_strategies() {
        let config = CrossModalConfig::default();
        let fusion_layer =
            FusionLayer::new(FusionStrategy::WeightedAverage, config.modality_weights);

        let mut embeddings = HashMap::new();
        embeddings.insert(Modality::Text, Vector::new(vec![1.0, 0.0, 0.0]));
        embeddings.insert(Modality::Image, Vector::new(vec![0.0, 1.0, 0.0]));

        let fused = fusion_layer.fuse(&embeddings).unwrap();
        assert_eq!(fused.dimensions, 3);
    }

    fn create_test_encoder(config: CrossModalConfig) -> CrossModalEncoder {
        let text_encoder = Box::new(MockTextEncoder::new(512));
        let image_encoder = Box::new(MockImageEncoder::new(512));
        let audio_encoder = Box::new(MockAudioEncoder::new(512));
        let video_encoder = Box::new(MockVideoEncoder::new(512));
        let graph_encoder = Box::new(MockGraphEncoder::new(512));

        CrossModalEncoder::new(
            config,
            text_encoder,
            image_encoder,
            audio_encoder,
            video_encoder,
            graph_encoder,
        )
    }
}
