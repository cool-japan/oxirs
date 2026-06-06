//! Cross-space transfer functions for Joint Embedding Spaces
//!
//! Zero-shot transfer, knowledge transfer, domain adaptation,
//! CLIP-style contrastive training, data augmentation, curriculum learning.

use super::joint_embedding_spaces_aligner::JointEmbeddingSpace;
use super::joint_embedding_spaces_types::{
    AudioAugmentation, ContrastiveOptimizer, ContrastivePairs, CurriculumLearning,
    DataAugmentation, DifficultySchedule, ImageAugmentation, JointEmbeddingConfig,
    LearningRateSchedule, PacingFunction, TextAugmentation,
};
use crate::cross_modal_embeddings::{
    AudioData, ImageData, Modality, ModalityData, MultiModalContent, VideoData,
};
use crate::Vector;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// CLIPAligner
// ─────────────────────────────────────────────────────────────────────────────

/// CLIP-style contrastive learning implementation
pub struct CLIPAligner {
    pub(crate) joint_space: JointEmbeddingSpace,
    pub(crate) optimizer: ContrastiveOptimizer,
    pub(crate) data_augmentation: DataAugmentation,
    pub(crate) curriculum: CurriculumLearning,
}

impl CLIPAligner {
    pub fn new(config: JointEmbeddingConfig) -> Self {
        let joint_space = JointEmbeddingSpace::new(config.clone());
        let optimizer = ContrastiveOptimizer::new(config.learning_rate, 0.9, config.weight_decay);
        let data_augmentation = DataAugmentation::default();
        let curriculum = CurriculumLearning::new();

        Self {
            joint_space,
            optimizer,
            data_augmentation,
            curriculum,
        }
    }

    /// Train CLIP-style alignment with contrastive learning
    pub fn train_alignment(
        &mut self,
        training_data: &[(MultiModalContent, MultiModalContent)],
        epochs: usize,
    ) -> Result<Vec<f32>> {
        let mut epoch_losses = Vec::new();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            for batch in training_data.chunks(self.joint_space.config.batch_size) {
                let (positive_pairs, negative_pairs) = self.create_contrastive_pairs(batch)?;

                let augmented_positive = self.augment_pairs(&positive_pairs)?;
                let augmented_negative = self.augment_pairs(&negative_pairs)?;

                let batch_loss = self
                    .joint_space
                    .contrastive_align(&augmented_positive, &augmented_negative)?;

                epoch_loss += batch_loss;
                batch_count += 1;

                if self.curriculum.enabled {
                    self.curriculum.update_difficulty(batch_loss);
                }
            }

            let avg_epoch_loss = epoch_loss / batch_count as f32;
            epoch_losses.push(avg_epoch_loss);

            self.optimizer.step_schedule();

            tracing::info!(
                "Epoch {}/{}: Average Loss = {:.4}, Temperature = {:.4}",
                epoch + 1,
                epochs,
                avg_epoch_loss,
                self.joint_space
                    .temperature_scheduler
                    .get_current_temperature()
            );
        }

        Ok(epoch_losses)
    }

    fn create_contrastive_pairs(
        &self,
        batch: &[(MultiModalContent, MultiModalContent)],
    ) -> Result<ContrastivePairs> {
        let mut positive_pairs = Vec::new();
        let mut negative_pairs = Vec::new();

        for (content1, content2) in batch {
            for (mod1, data1) in &content1.modalities {
                for (mod2, data2) in &content2.modalities {
                    if let (Ok(emb1), Ok(emb2)) = (
                        self.extract_embedding(*mod1, data1),
                        self.extract_embedding(*mod2, data2),
                    ) {
                        positive_pairs.push((*mod1, emb1, *mod2, emb2));
                    }
                }
            }
        }

        let batch_size = batch.len();
        for i in 0..batch_size {
            for j in 0..batch_size {
                if i != j {
                    let (content1, _) = &batch[i];
                    let (_, content2) = &batch[j];

                    for (mod1, data1) in &content1.modalities {
                        for (mod2, data2) in &content2.modalities {
                            if let (Ok(emb1), Ok(emb2)) = (
                                self.extract_embedding(*mod1, data1),
                                self.extract_embedding(*mod2, data2),
                            ) {
                                negative_pairs.push((*mod1, emb1, *mod2, emb2));
                            }
                        }
                    }
                }
            }
        }

        let max_negatives = positive_pairs.len() * self.joint_space.config.negative_samples;
        negative_pairs.truncate(max_negatives);

        Ok((positive_pairs, negative_pairs))
    }

    fn extract_embedding(&self, modality: Modality, data: &ModalityData) -> Result<Vector> {
        match (modality, data) {
            (Modality::Text, ModalityData::Text(text)) => {
                let words: Vec<&str> = text.split_whitespace().collect();
                let embedding = self.create_text_embedding(&words);
                Ok(embedding)
            }
            (Modality::Image, ModalityData::Image(image)) => {
                let embedding = self.create_image_embedding(image);
                Ok(embedding)
            }
            (Modality::Audio, ModalityData::Audio(audio)) => {
                let embedding = self.create_audio_embedding(audio);
                Ok(embedding)
            }
            (Modality::Video, ModalityData::Video(video)) => {
                let embedding = self.create_video_embedding(video);
                Ok(embedding)
            }
            (Modality::Numeric, ModalityData::Numeric(values)) => Ok(Vector::new(values.clone())),
            _ => Err(anyhow!("Modality-data type mismatch")),
        }
    }

    pub(crate) fn create_text_embedding(&self, words: &[&str]) -> Vector {
        let mut embedding = vec![0.0; 768];

        for (i, word) in words.iter().enumerate().take(100) {
            let hash = self.simple_hash(word) as usize;
            let idx = hash % embedding.len();
            embedding[idx] += 1.0 / (i + 1) as f32;
        }

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut embedding {
                *value /= norm;
            }
        }

        Vector::new(embedding)
    }

    fn create_image_embedding(&self, image: &ImageData) -> Vector {
        let mut embedding = vec![0.0; 2048];

        let color_features = self.extract_color_features(image);
        for (i, &feature) in color_features.iter().enumerate().take(256) {
            if i < embedding.len() {
                embedding[i] = feature;
            }
        }

        let texture_features = self.extract_texture_features(image);
        for (i, &feature) in texture_features.iter().enumerate().take(256) {
            if i + 256 < embedding.len() {
                embedding[i + 256] = feature;
            }
        }

        Vector::new(embedding)
    }

    fn create_audio_embedding(&self, audio: &AudioData) -> Vector {
        let mut embedding = vec![0.0; 1024];

        if let Some(ref features) = audio.features {
            for (i, &feature) in features.iter().enumerate().take(embedding.len()) {
                embedding[i] = feature;
            }
        } else {
            let spectral_features = self.extract_spectral_features(audio);
            for (i, &feature) in spectral_features.iter().enumerate().take(embedding.len()) {
                embedding[i] = feature;
            }
        }

        Vector::new(embedding)
    }

    fn create_video_embedding(&self, video: &VideoData) -> Vector {
        let mut embedding = vec![0.0; 1536];

        if !video.frames.is_empty() {
            let frame_embedding = self.create_image_embedding(&video.frames[0]);
            let frame_values = frame_embedding.as_f32();
            for (i, &value) in frame_values.iter().enumerate().take(1024) {
                if i < embedding.len() {
                    embedding[i] = value;
                }
            }
        }

        if let Some(ref audio) = video.audio {
            let audio_embedding = self.create_audio_embedding(audio);
            let audio_values = audio_embedding.as_f32();
            for (i, &value) in audio_values.iter().enumerate().take(512) {
                if i + 1024 < embedding.len() {
                    embedding[i + 1024] = value;
                }
            }
        }

        Vector::new(embedding)
    }

    fn simple_hash(&self, text: &str) -> u64 {
        let mut hash = 5381u64;
        for byte in text.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash
    }

    fn extract_color_features(&self, image: &ImageData) -> Vec<f32> {
        let mut histogram = vec![0.0; 256];

        match image.format {
            crate::cross_modal_embeddings::ImageFormat::RGB => {
                for chunk in image.data.chunks(3) {
                    if chunk.len() == 3 {
                        let intensity = (chunk[0] as f32 + chunk[1] as f32 + chunk[2] as f32) / 3.0;
                        let bin = (intensity as usize).min(255);
                        histogram[bin] += 1.0;
                    }
                }
            }
            _ => {
                for &pixel in &image.data {
                    let bin = (pixel as usize).min(255);
                    histogram[bin] += 1.0;
                }
            }
        }

        let total: f32 = histogram.iter().sum();
        if total > 0.0 {
            for value in &mut histogram {
                *value /= total;
            }
        }

        histogram
    }

    fn extract_texture_features(&self, image: &ImageData) -> Vec<f32> {
        let mut features = vec![0.0; 256];

        let width = image.width as usize;
        let height = image.height as usize;

        if width > 2 && height > 2 {
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let center_idx = y * width + x;
                    if center_idx < image.data.len() {
                        let center = image.data[center_idx];
                        let mut pattern = 0u8;

                        let neighbors = [
                            (-1i32, -1i32),
                            (0, -1),
                            (1, -1),
                            (-1, 0),
                            (1, 0),
                            (-1, 1),
                            (0, 1),
                            (1, 1),
                        ];

                        for (bit, (dx, dy)) in neighbors.iter().enumerate() {
                            let nx = (x as i32 + dx) as usize;
                            let ny = (y as i32 + dy) as usize;
                            let neighbor_idx = ny * width + nx;

                            if neighbor_idx < image.data.len() && image.data[neighbor_idx] > center
                            {
                                pattern |= 1 << bit;
                            }
                        }

                        features[pattern as usize] += 1.0;
                    }
                }
            }
        }

        let total: f32 = features.iter().sum();
        if total > 0.0 {
            for value in &mut features {
                *value /= total;
            }
        }

        features
    }

    fn extract_spectral_features(&self, audio: &AudioData) -> Vec<f32> {
        let mut features = vec![0.0; 128];

        if !audio.samples.is_empty() {
            let chunk_size = audio.samples.len() / features.len();

            for (i, feature) in features.iter_mut().enumerate() {
                let start = i * chunk_size;
                let end = ((i + 1) * chunk_size).min(audio.samples.len());

                if start < end {
                    let chunk = &audio.samples[start..end];
                    let energy: f32 = chunk.iter().map(|x| x * x).sum();
                    *feature = energy.sqrt() / (chunk.len() as f32).sqrt();
                }
            }
        }

        features
    }

    fn augment_pairs(
        &self,
        pairs: &[(Modality, Vector, Modality, Vector)],
    ) -> Result<Vec<(Modality, Vector, Modality, Vector)>> {
        let mut augmented = Vec::new();

        for (mod1, emb1, mod2, emb2) in pairs {
            let aug_emb1 = self.add_noise(emb1, 0.01)?;
            let aug_emb2 = self.add_noise(emb2, 0.01)?;
            augmented.push((*mod1, aug_emb1, *mod2, aug_emb2));
        }

        Ok(augmented)
    }

    fn add_noise(&self, embedding: &Vector, noise_std: f32) -> Result<Vector> {
        let values = embedding.as_f32();
        let mut noisy_values = Vec::with_capacity(values.len());

        for (i, &value) in values.iter().enumerate() {
            let noise = ((i as f32 * 0.1234).sin() * noise_std).clamp(-0.1, 0.1);
            noisy_values.push(value + noise);
        }

        Ok(Vector::new(noisy_values))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ContrastiveOptimizer impl
// ─────────────────────────────────────────────────────────────────────────────

impl ContrastiveOptimizer {
    pub fn new(learning_rate: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            gradient_history: HashMap::new(),
            adaptive_lr: true,
            lr_schedule: LearningRateSchedule::CosineAnnealing {
                min_lr: learning_rate * 0.01,
                max_epochs: 100,
            },
        }
    }

    pub fn step_schedule(&mut self) {
        match self.lr_schedule {
            LearningRateSchedule::StepDecay {
                step_size: _,
                gamma,
            } => {
                self.learning_rate *= gamma;
            }
            LearningRateSchedule::ExponentialDecay { gamma } => {
                self.learning_rate *= gamma;
            }
            LearningRateSchedule::CosineAnnealing {
                min_lr,
                max_epochs: _,
            } => {
                let progress = 0.01;
                let lr_range = self.learning_rate - min_lr;
                self.learning_rate =
                    min_lr + lr_range * (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0;
            }
            LearningRateSchedule::Constant => {}
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DataAugmentation impl
// ─────────────────────────────────────────────────────────────────────────────

impl Default for DataAugmentation {
    fn default() -> Self {
        Self {
            text_augmentations: vec![
                TextAugmentation::RandomWordDropout(0.1),
                TextAugmentation::SynonymReplacement(0.1),
            ],
            image_augmentations: vec![
                ImageAugmentation::RandomFlip {
                    horizontal: true,
                    vertical: false,
                },
                ImageAugmentation::ColorJitter {
                    brightness: 0.2,
                    contrast: 0.2,
                    saturation: 0.2,
                },
            ],
            audio_augmentations: vec![
                AudioAugmentation::AddNoise { snr_db: 20.0 },
                AudioAugmentation::TimeStretch { factor: 1.1 },
            ],
            cross_modal_mixup: false,
            augmentation_probability: 0.5,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CurriculumLearning impl
// ─────────────────────────────────────────────────────────────────────────────

impl Default for CurriculumLearning {
    fn default() -> Self {
        Self::new()
    }
}

impl CurriculumLearning {
    pub fn new() -> Self {
        Self {
            enabled: false,
            current_difficulty: 0.0,
            difficulty_schedule: DifficultySchedule::Linear {
                start: 0.0,
                end: 1.0,
                epochs: 50,
            },
            pacing_function: PacingFunction::Root,
            competence_threshold: 0.8,
        }
    }

    pub fn update_difficulty(&mut self, loss: f32) {
        if self.enabled {
            if loss < self.competence_threshold {
                self.current_difficulty = (self.current_difficulty + 0.01).min(1.0);
            } else {
                self.current_difficulty = (self.current_difficulty - 0.005).max(0.0);
            }
        }
    }
}

/// Zero-shot cross-modal retrieval helper (delegates to JointEmbeddingSpace)
pub fn zero_shot_retrieval(
    space: &JointEmbeddingSpace,
    query_modality: Modality,
    query_embedding: &Vector,
    target_modality: Modality,
    target_embeddings: &[Vector],
    top_k: usize,
) -> Result<Vec<(usize, f32)>> {
    space.cross_modal_search(
        query_modality,
        query_embedding,
        target_modality,
        target_embeddings,
        top_k,
    )
}
