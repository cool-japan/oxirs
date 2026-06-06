//! Alignment algorithms for Joint Embedding Spaces
//!
//! CCA, Procrustes, manifold alignment, cross-space distance computation.

use super::joint_embedding_spaces_types::{
    ActivationFunction, AlignmentPair, CrossModalAttention, DomainAdapter, DomainStatistics,
    JointEmbeddingConfig, LinearProjector, ScheduleType, TemperatureScheduler, TrainingStatistics,
};
use crate::{cross_modal_embeddings::Modality, Vector};
use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// LinearProjector impl
// ─────────────────────────────────────────────────────────────────────────────

impl LinearProjector {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        dropout_rate: f32,
        activation: ActivationFunction,
    ) -> Self {
        // Xavier/Glorot initialization
        let limit = (6.0 / (input_dim + output_dim) as f32).sqrt();
        let mut weights = Vec::with_capacity(output_dim);

        for _ in 0..output_dim {
            let mut row = Vec::with_capacity(input_dim);
            for _ in 0..input_dim {
                let weight = ((row.len() as f32 * 0.01) % 2.0 - 1.0) * limit;
                row.push(weight);
            }
            weights.push(row);
        }

        let bias = vec![0.0; output_dim];

        Self {
            weights,
            bias,
            input_dim,
            output_dim,
            dropout_rate,
            activation,
        }
    }

    pub fn forward(&self, input: &Vector) -> Result<Vector> {
        if input.dimensions != self.input_dim {
            return Err(anyhow!(
                "Input dimension mismatch: expected {}, got {}",
                self.input_dim,
                input.dimensions
            ));
        }

        let input_values = input.as_f32();
        let mut output = vec![0.0; self.output_dim];

        for (i, output_val) in output.iter_mut().enumerate().take(self.output_dim) {
            let mut sum = self.bias[i];
            for (j, &input_val) in input_values.iter().enumerate().take(self.input_dim) {
                sum += input_val * self.weights[i][j];
            }
            *output_val = sum;
        }

        for value in &mut output {
            *value = self.apply_activation(*value);
        }

        if self.dropout_rate > 0.0 {
            for (i, value) in output.iter_mut().enumerate() {
                if (i as f32 * 0.12345) % 1.0 < self.dropout_rate {
                    *value = 0.0;
                } else {
                    *value /= 1.0 - self.dropout_rate;
                }
            }
        }

        Ok(Vector::new(output))
    }

    fn apply_activation(&self, x: f32) -> f32 {
        match self.activation {
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::GELU => {
                let sqrt_2_pi = (2.0 / std::f32::consts::PI).sqrt();
                let inner = sqrt_2_pi * (x + 0.044715 * x.powi(3));
                0.5 * x * (1.0 + inner.tanh())
            }
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Swish => x * (1.0 / (1.0 + (-x).exp())),
            ActivationFunction::Mish => x * (1.0 + x.exp()).ln().tanh(),
            ActivationFunction::LeakyReLU(alpha) => {
                if x > 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
        }
    }

    pub fn update_weights(&mut self, gradients: &[Vec<f32>], learning_rate: f32) {
        for i in 0..self.output_dim {
            for j in 0..self.input_dim {
                if i < gradients.len() && j < gradients[i].len() {
                    self.weights[i][j] -= learning_rate * gradients[i][j];
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CrossModalAttention impl
// ─────────────────────────────────────────────────────────────────────────────

impl CrossModalAttention {
    pub fn new(
        input_dim: usize,
        num_heads: usize,
        dropout_rate: f32,
        enable_relative_pos: bool,
    ) -> Self {
        let head_dim = input_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        Self {
            query_projector: LinearProjector::new(
                input_dim,
                input_dim,
                dropout_rate,
                ActivationFunction::ReLU,
            ),
            key_projector: LinearProjector::new(
                input_dim,
                input_dim,
                dropout_rate,
                ActivationFunction::ReLU,
            ),
            value_projector: LinearProjector::new(
                input_dim,
                input_dim,
                dropout_rate,
                ActivationFunction::ReLU,
            ),
            output_projector: LinearProjector::new(
                input_dim,
                input_dim,
                dropout_rate,
                ActivationFunction::ReLU,
            ),
            num_heads,
            head_dim,
            dropout_rate,
            scale,
            enable_relative_pos,
        }
    }

    pub fn cross_attention(
        &self,
        query_modality: &Vector,
        key_modality: &Vector,
        value_modality: &Vector,
    ) -> Result<Vector> {
        let query = self.query_projector.forward(query_modality)?;
        let key = self.key_projector.forward(key_modality)?;
        let value = self.value_projector.forward(value_modality)?;

        let attended = self.multi_head_attention(&query, &key, &value)?;
        self.output_projector.forward(&attended)
    }

    fn multi_head_attention(&self, query: &Vector, key: &Vector, value: &Vector) -> Result<Vector> {
        let query_vals = query.as_f32();
        let key_vals = key.as_f32();
        let value_vals = value.as_f32();

        if query_vals.len() != key_vals.len() || key_vals.len() != value_vals.len() {
            return Err(anyhow!("Dimension mismatch in attention"));
        }

        let _seq_len = query_vals.len() / self.head_dim;
        let mut output = vec![0.0; query_vals.len()];

        for head in 0..self.num_heads {
            let head_start = head * self.head_dim;
            let head_end = head_start + self.head_dim;

            let head_query = &query_vals[head_start..head_end];
            let head_key = &key_vals[head_start..head_end];
            let head_value = &value_vals[head_start..head_end];

            let attention_score = self.compute_attention_score(head_query, head_key);

            for i in 0..self.head_dim {
                output[head_start + i] = head_value[i] * attention_score;
            }
        }

        if self.enable_relative_pos {
            self.apply_relative_position_encoding(&mut output)?;
        }

        Ok(Vector::new(output))
    }

    fn compute_attention_score(&self, query: &[f32], key: &[f32]) -> f32 {
        let dot_product: f32 = query.iter().zip(key.iter()).map(|(q, k)| q * k).sum();
        let scaled_score = dot_product * self.scale;
        scaled_score.tanh()
    }

    fn apply_relative_position_encoding(&self, output: &mut [f32]) -> Result<()> {
        let output_len = output.len();
        for (i, value) in output.iter_mut().enumerate() {
            let pos_encoding = (i as f32 / output_len as f32).sin();
            *value += 0.1 * pos_encoding;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TemperatureScheduler impl
// ─────────────────────────────────────────────────────────────────────────────

impl TemperatureScheduler {
    pub fn new(
        initial_temperature: f32,
        final_temperature: f32,
        decay_steps: usize,
        schedule_type: ScheduleType,
    ) -> Self {
        Self {
            initial_temperature,
            final_temperature,
            decay_steps,
            current_step: 0,
            schedule_type,
        }
    }

    pub fn get_current_temperature(&self) -> f32 {
        if self.current_step >= self.decay_steps {
            return self.final_temperature;
        }

        let progress = self.current_step as f32 / self.decay_steps as f32;

        match self.schedule_type {
            ScheduleType::Linear => {
                self.initial_temperature
                    + (self.final_temperature - self.initial_temperature) * progress
            }
            ScheduleType::Exponential => {
                self.initial_temperature
                    * (self.final_temperature / self.initial_temperature).powf(progress)
            }
            ScheduleType::Cosine => {
                let cosine_progress = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
                self.final_temperature
                    + (self.initial_temperature - self.final_temperature) * cosine_progress
            }
            ScheduleType::Warmup => {
                if progress < 0.1 {
                    self.initial_temperature * (progress / 0.1)
                } else {
                    let decay_progress = (progress - 0.1) / 0.9;
                    self.initial_temperature
                        + (self.final_temperature - self.initial_temperature) * decay_progress
                }
            }
        }
    }

    pub fn step(&mut self) {
        self.current_step += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DomainAdapter impl
// ─────────────────────────────────────────────────────────────────────────────

impl DomainAdapter {
    pub fn new(adaptation_strength: f32) -> Self {
        Self {
            source_stats: DomainStatistics::default(),
            target_stats: DomainStatistics::default(),
            adaptation_weights: Vec::new(),
            domain_classifier: None,
            adaptation_strength,
        }
    }

    pub fn adapt_embedding(&self, embedding: &Vector, is_source_domain: bool) -> Result<Vector> {
        let input_values = embedding.as_f32();
        let mut adapted_values = input_values.clone();

        if self.adaptation_weights.len() != input_values.len() {
            return Ok(embedding.clone());
        }

        let stats = if is_source_domain {
            &self.source_stats
        } else {
            &self.target_stats
        };

        for (i, adapted_value) in adapted_values.iter_mut().enumerate() {
            if i < stats.mean.len() && i < stats.variance.len() {
                let normalized =
                    (*adapted_value - stats.mean[i]) / (stats.variance[i].sqrt() + 1e-8);
                *adapted_value = normalized * self.adaptation_weights[i] * self.adaptation_strength
                    + *adapted_value * (1.0 - self.adaptation_strength);
            }
        }

        Ok(Vector::new(adapted_values))
    }

    pub fn update_domain_statistics(&mut self, embeddings: &[Vector], is_source_domain: bool) {
        let stats = if is_source_domain {
            &mut self.source_stats
        } else {
            &mut self.target_stats
        };

        if embeddings.is_empty() {
            return;
        }

        let dim = embeddings[0].dimensions;
        if stats.mean.len() != dim {
            stats.mean = vec![0.0; dim];
            stats.variance = vec![0.0; dim];
            stats.sample_count = 0;
        }

        for embedding in embeddings {
            let values = embedding.as_f32();
            for (i, &value) in values.iter().enumerate().take(dim) {
                let delta = value - stats.mean[i];
                stats.sample_count += 1;
                stats.mean[i] += delta / stats.sample_count as f32;
                let delta2 = value - stats.mean[i];
                stats.variance[i] += delta * delta2;
            }
        }

        if stats.sample_count > 1 {
            for variance in &mut stats.variance {
                *variance /= (stats.sample_count - 1) as f32;
            }
        }

        self.update_adaptation_weights();
    }

    fn update_adaptation_weights(&mut self) {
        let dim = self.source_stats.mean.len();
        if dim == 0 || dim != self.target_stats.mean.len() {
            return;
        }

        self.adaptation_weights = vec![1.0; dim];

        for i in 0..dim {
            let mean_diff = (self.source_stats.mean[i] - self.target_stats.mean[i]).abs();
            let var_ratio = (self.source_stats.variance[i]
                / (self.target_stats.variance[i] + 1e-8))
                .ln()
                .abs();

            let discrepancy = mean_diff + 0.5 * var_ratio;
            self.adaptation_weights[i] = 1.0 / (1.0 + discrepancy);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JointEmbeddingSpace
// ─────────────────────────────────────────────────────────────────────────────

/// Joint embedding space for cross-modal alignment
pub struct JointEmbeddingSpace {
    pub(crate) config: JointEmbeddingConfig,
    pub(crate) text_projector: LinearProjector,
    pub(crate) image_projector: LinearProjector,
    pub(crate) audio_projector: LinearProjector,
    pub(crate) video_projector: LinearProjector,
    pub(crate) attention_mechanism: CrossModalAttention,
    pub(crate) alignment_cache: Arc<RwLock<HashMap<String, AlignmentPair>>>,
    pub(crate) training_stats: Arc<RwLock<TrainingStatistics>>,
    pub(crate) temperature_scheduler: TemperatureScheduler,
    pub(crate) domain_adapter: DomainAdapter,
}

impl JointEmbeddingSpace {
    pub fn new(config: JointEmbeddingConfig) -> Self {
        let text_projector =
            LinearProjector::new(768, config.joint_dim, 0.1, ActivationFunction::GELU);

        let image_projector =
            LinearProjector::new(2048, config.joint_dim, 0.1, ActivationFunction::GELU);

        let audio_projector =
            LinearProjector::new(1024, config.joint_dim, 0.1, ActivationFunction::GELU);

        let video_projector =
            LinearProjector::new(1536, config.joint_dim, 0.1, ActivationFunction::GELU);

        let attention_mechanism = CrossModalAttention::new(config.joint_dim, 8, 0.1, true);

        let temperature_scheduler = TemperatureScheduler::new(
            config.temperature * 2.0,
            config.temperature,
            1000,
            ScheduleType::Cosine,
        );

        let domain_adapter = DomainAdapter::new(config.alignment_strength);

        Self {
            config,
            text_projector,
            image_projector,
            audio_projector,
            video_projector,
            attention_mechanism,
            alignment_cache: Arc::new(RwLock::new(HashMap::new())),
            training_stats: Arc::new(RwLock::new(TrainingStatistics::default())),
            temperature_scheduler,
            domain_adapter,
        }
    }

    /// Project modality-specific embedding to joint space
    pub fn project_to_joint_space(&self, modality: Modality, embedding: &Vector) -> Result<Vector> {
        let projected = match modality {
            Modality::Text => self.text_projector.forward(embedding)?,
            Modality::Image => self.image_projector.forward(embedding)?,
            Modality::Audio => self.audio_projector.forward(embedding)?,
            Modality::Video => self.video_projector.forward(embedding)?,
            _ => self.text_projector.forward(embedding)?,
        };

        Ok(projected.normalized())
    }

    /// Compute cross-modal similarity in joint space
    pub fn cross_modal_similarity(
        &self,
        modality1: Modality,
        embedding1: &Vector,
        modality2: Modality,
        embedding2: &Vector,
    ) -> Result<f32> {
        let joint_emb1 = self.project_to_joint_space(modality1, embedding1)?;
        let joint_emb2 = self.project_to_joint_space(modality2, embedding2)?;

        if modality1 != modality2 {
            let attended_emb1 =
                self.attention_mechanism
                    .cross_attention(&joint_emb1, &joint_emb2, &joint_emb2)?;
            let attended_emb2 =
                self.attention_mechanism
                    .cross_attention(&joint_emb2, &joint_emb1, &joint_emb1)?;

            attended_emb1.cosine_similarity(&attended_emb2)
        } else {
            joint_emb1.cosine_similarity(&joint_emb2)
        }
    }

    /// Contrastive learning alignment training
    pub fn contrastive_align(
        &mut self,
        positive_pairs: &[(Modality, Vector, Modality, Vector)],
        negative_pairs: &[(Modality, Vector, Modality, Vector)],
    ) -> Result<f32> {
        let mut total_loss = 0.0;
        let temperature = self.temperature_scheduler.get_current_temperature();

        for (mod1, emb1, mod2, emb2) in positive_pairs {
            let similarity = self.cross_modal_similarity(*mod1, emb1, *mod2, emb2)?;
            let positive_score = similarity / temperature;
            let positive_loss = -positive_score.ln_1p();
            total_loss += positive_loss;

            self.cache_alignment(*mod1, emb1.clone(), *mod2, emb2.clone(), similarity);
        }

        for (mod1, emb1, mod2, emb2) in negative_pairs {
            let similarity = self.cross_modal_similarity(*mod1, emb1, *mod2, emb2)?;
            let negative_score = similarity / temperature;
            let negative_loss = (negative_score + self.config.margin).max(0.0);
            total_loss += negative_loss;
        }

        self.update_training_stats(positive_pairs.len(), negative_pairs.len(), total_loss);
        self.temperature_scheduler.step();

        Ok(total_loss / (positive_pairs.len() + negative_pairs.len()) as f32)
    }

    /// Zero-shot cross-modal retrieval
    pub fn zero_shot_retrieval(
        &self,
        query_modality: Modality,
        query_embedding: &Vector,
        target_modality: Modality,
        target_embeddings: &[Vector],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        // Project query to joint space
        let _query_joint = self.project_to_joint_space(query_modality, query_embedding)?;

        // Search across target modality
        self.cross_modal_search(
            query_modality,
            query_embedding,
            target_modality,
            target_embeddings,
            top_k,
        )
    }

    /// Find cross-modal nearest neighbors in joint space
    pub fn cross_modal_search(
        &self,
        query_modality: Modality,
        query_embedding: &Vector,
        candidate_modality: Modality,
        candidate_embeddings: &[Vector],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let query_joint = self.project_to_joint_space(query_modality, query_embedding)?;
        let mut similarities = Vec::new();

        for (idx, candidate) in candidate_embeddings.iter().enumerate() {
            let candidate_joint = self.project_to_joint_space(candidate_modality, candidate)?;

            let similarity = if query_modality != candidate_modality {
                let attended_query = self.attention_mechanism.cross_attention(
                    &query_joint,
                    &candidate_joint,
                    &candidate_joint,
                )?;
                attended_query.cosine_similarity(&candidate_joint)?
            } else {
                query_joint.cosine_similarity(&candidate_joint)?
            };

            similarities.push((idx, similarity));
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);

        Ok(similarities)
    }

    /// Multi-modal fusion in joint space
    pub fn multi_modal_fusion(&self, modalities: &[(Modality, Vector)]) -> Result<Vector> {
        if modalities.is_empty() {
            return Err(anyhow!("No modalities provided for fusion"));
        }

        let mut joint_embeddings = Vec::new();
        for (modality, embedding) in modalities {
            let joint_emb = self.project_to_joint_space(*modality, embedding)?;
            joint_embeddings.push(joint_emb);
        }

        let mut attended_embeddings = Vec::new();
        for i in 0..joint_embeddings.len() {
            let mut attended = joint_embeddings[i].clone();

            for j in 0..joint_embeddings.len() {
                if i != j {
                    let cross_attended = self.attention_mechanism.cross_attention(
                        &joint_embeddings[i],
                        &joint_embeddings[j],
                        &joint_embeddings[j],
                    )?;

                    let weight = 1.0 / joint_embeddings.len() as f32;
                    attended = attended.add(&cross_attended.scale(weight))?;
                }
            }

            attended_embeddings.push(attended);
        }

        if attended_embeddings.len() == 1 {
            Ok(attended_embeddings[0].clone())
        } else {
            let mut fused = attended_embeddings[0].clone();
            for embedding in attended_embeddings.iter().skip(1) {
                fused = fused.add(embedding)?;
            }
            Ok(fused.scale(1.0 / attended_embeddings.len() as f32))
        }
    }

    pub(crate) fn cache_alignment(
        &self,
        mod1: Modality,
        emb1: Vector,
        mod2: Modality,
        emb2: Vector,
        similarity: f32,
    ) {
        let alignment = AlignmentPair {
            modality1: mod1,
            modality2: mod2,
            embedding1: emb1,
            embedding2: emb2,
            similarity,
            confidence: similarity.abs(),
            timestamp: std::time::SystemTime::now(),
        };

        let cache_key = format!("{mod1:?}_{mod2:?}_{similarity}");
        let mut cache = self.alignment_cache.write();
        cache.insert(cache_key, alignment);

        if cache.len() > 10000 {
            let mut entries: Vec<_> = cache.iter().collect();
            entries.sort_by_key(|(_, v)| v.timestamp);
            let oldest_key = entries[0].0.clone();
            cache.remove(&oldest_key);
        }
    }

    pub(crate) fn update_training_stats(
        &self,
        positive_count: usize,
        negative_count: usize,
        loss: f32,
    ) {
        let mut stats = self.training_stats.write();
        stats.total_samples += (positive_count + negative_count) as u64;
        stats.positive_pairs += positive_count as u64;
        stats.negative_pairs += negative_count as u64;

        let total_samples = stats.total_samples as f32;
        stats.average_loss = (stats.average_loss * (total_samples - 1.0) + loss) / total_samples;
    }

    /// Get training statistics
    pub fn get_training_stats(&self) -> TrainingStatistics {
        self.training_stats.read().clone()
    }

    /// Get alignment cache statistics
    pub fn get_cache_stats(&self) -> (usize, f32) {
        let cache = self.alignment_cache.read();
        let cache_size = cache.len();
        let avg_similarity = if cache.is_empty() {
            0.0
        } else {
            cache.values().map(|a| a.similarity).sum::<f32>() / cache_size as f32
        };
        (cache_size, avg_similarity)
    }

    /// Evaluate cross-modal retrieval performance
    pub fn evaluate_retrieval(
        &self,
        test_pairs: &[(Modality, Vector, Modality, Vector)],
        distractors: &[(Modality, Vector)],
        k_values: &[usize],
    ) -> Result<HashMap<usize, f32>> {
        let mut recall_at_k = HashMap::new();

        for &k in k_values {
            let mut total_recall = 0.0;

            for (query_mod, query_emb, target_mod, target_emb) in test_pairs {
                let mut candidates = vec![target_emb.clone()];
                for (distractor_mod, distractor_emb) in distractors {
                    if *distractor_mod == *target_mod {
                        candidates.push(distractor_emb.clone());
                    }
                }

                let results =
                    self.cross_modal_search(*query_mod, query_emb, *target_mod, &candidates, k)?;

                let found_target = results.iter().any(|(idx, _)| *idx == 0);
                if found_target {
                    total_recall += 1.0;
                }
            }

            recall_at_k.insert(k, total_recall / test_pairs.len() as f32);
        }

        Ok(recall_at_k)
    }
}
