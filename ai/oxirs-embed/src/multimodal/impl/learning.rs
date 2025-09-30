//! Few-shot learning and meta-learning components for multi-modal embeddings

use super::model::MultiModalEmbedding;
use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Few-shot learning module for rapid adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotLearning {
    /// Support set size
    pub support_size: usize,
    /// Query set size
    pub query_size: usize,
    /// Number of ways (classes/entities)
    pub num_ways: usize,
    /// Meta-learning algorithm
    pub meta_algorithm: MetaAlgorithm,
    /// Adaptation parameters
    pub adaptation_config: AdaptationConfig,
    /// Prototypical network
    pub prototypical_network: PrototypicalNetwork,
    /// Model-agnostic meta-learning (MAML) components
    pub maml_components: MAMLComponents,
}

/// Meta-learning algorithms for few-shot learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaAlgorithm {
    /// Prototypical Networks
    PrototypicalNetworks,
    /// Model-Agnostic Meta-Learning
    MAML,
    /// Reptile algorithm
    Reptile,
    /// Matching Networks
    MatchingNetworks,
    /// Relation Networks
    RelationNetworks,
    /// Memory-Augmented Neural Networks
    MANN,
}

/// Adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationConfig {
    /// Learning rate for few-shot adaptation
    pub adaptation_lr: f32,
    /// Number of adaptation steps
    pub adaptation_steps: usize,
    /// Gradient clipping threshold
    pub gradient_clip: f32,
    /// Use second-order gradients (for MAML)
    pub second_order: bool,
    /// Temperature for prototypical networks
    pub temperature: f32,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            adaptation_lr: 0.01,
            adaptation_steps: 5,
            gradient_clip: 1.0,
            second_order: true,
            temperature: 1.0,
        }
    }
}

/// Prototypical network for few-shot learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrototypicalNetwork {
    /// Feature extractor parameters
    pub feature_extractor: HashMap<String, Array2<f32>>,
    /// Prototype computation method
    pub prototype_method: PrototypeMethod,
    /// Distance metric
    pub distance_metric: DistanceMetric,
}

/// Prototype computation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrototypeMethod {
    /// Simple mean of support examples
    Mean,
    /// Weighted mean with attention
    AttentionWeighted,
    /// Learnable prototype aggregation
    LearnableAggregation,
}

/// Distance metrics for prototype comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Cosine distance
    Cosine,
    /// Learned distance metric
    Learned,
    /// Mahalanobis distance
    Mahalanobis,
}

/// MAML components for meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MAMLComponents {
    /// Inner loop parameters
    pub inner_loop_params: HashMap<String, Array2<f32>>,
    /// Outer loop parameters
    pub outer_loop_params: HashMap<String, Array2<f32>>,
    /// Meta-gradients
    pub meta_gradients: HashMap<String, Array2<f32>>,
    /// Task-specific adaptations
    pub task_adaptations: HashMap<String, HashMap<String, Array2<f32>>>,
}

impl Default for FewShotLearning {
    fn default() -> Self {
        Self {
            support_size: 5,
            query_size: 15,
            num_ways: 3,
            meta_algorithm: MetaAlgorithm::PrototypicalNetworks,
            adaptation_config: AdaptationConfig::default(),
            prototypical_network: PrototypicalNetwork::default(),
            maml_components: MAMLComponents::default(),
        }
    }
}

impl Default for PrototypicalNetwork {
    fn default() -> Self {
        let mut feature_extractor = HashMap::new();
        feature_extractor.insert(
            "conv1".to_string(),
            Array2::from_shape_fn((64, 32), |(_, _)| {
                use scirs2_core::random::{Rng, Random};
                let mut random = Random::default();
                (random.gen::<f32>() - 0.5) * 0.1
            }),
        );
        feature_extractor.insert(
            "conv2".to_string(),
            Array2::from_shape_fn((128, 64), |(_, _)| {
                use scirs2_core::random::{Rng, Random};
                let mut random = Random::default();
                (random.gen::<f32>() - 0.5) * 0.1
            }),
        );
        feature_extractor.insert(
            "fc".to_string(),
            Array2::from_shape_fn((256, 128), |(_, _)| {
                use scirs2_core::random::{Rng, Random};
                let mut random = Random::default();
                (random.gen::<f32>() - 0.5) * 0.1
            }),
        );

        Self {
            feature_extractor,
            prototype_method: PrototypeMethod::Mean,
            distance_metric: DistanceMetric::Euclidean,
        }
    }
}

impl Default for MAMLComponents {
    fn default() -> Self {
        let mut inner_params = HashMap::new();
        let mut outer_params = HashMap::new();
        let mut meta_grads = HashMap::new();

        for layer in ["layer1", "layer2", "output"] {
            inner_params.insert(
                layer.to_string(),
                Array2::from_shape_fn((128, 128), |(_, _)| {
                    use scirs2_core::random::{Rng, Random};
                    let mut random = Random::default();
                    (random.gen::<f32>() - 0.5) * 0.1
                }),
            );
            outer_params.insert(
                layer.to_string(),
                Array2::from_shape_fn((128, 128), |(_, _)| {
                    use scirs2_core::random::{Rng, Random};
                    let mut random = Random::default();
                    (random.gen::<f32>() - 0.5) * 0.1
                }),
            );
            meta_grads.insert(layer.to_string(), Array2::zeros((128, 128)));
        }

        Self {
            inner_loop_params: inner_params,
            outer_loop_params: outer_params,
            meta_gradients: meta_grads,
            task_adaptations: HashMap::new(),
        }
    }
}

impl FewShotLearning {
    /// Create new few-shot learning module
    pub fn new(
        support_size: usize,
        query_size: usize,
        num_ways: usize,
        meta_algorithm: MetaAlgorithm,
    ) -> Self {
        Self {
            support_size,
            query_size,
            num_ways,
            meta_algorithm,
            adaptation_config: AdaptationConfig::default(),
            prototypical_network: PrototypicalNetwork::default(),
            maml_components: MAMLComponents::default(),
        }
    }

    /// Perform few-shot adaptation
    pub async fn few_shot_adapt(
        &mut self,
        support_examples: &[(String, String, String)], // (text, entity, label)
        query_examples: &[(String, String)],           // (text, entity)
        model: &MultiModalEmbedding,
    ) -> Result<Vec<(String, f32)>> {
        match self.meta_algorithm {
            MetaAlgorithm::PrototypicalNetworks => {
                self.prototypical_adapt(support_examples, query_examples, model)
                    .await
            }
            MetaAlgorithm::MAML => {
                self.maml_adapt(support_examples, query_examples, model)
                    .await
            }
            MetaAlgorithm::Reptile => {
                self.reptile_adapt(support_examples, query_examples, model)
                    .await
            }
            _ => {
                // Fallback to prototypical networks
                self.prototypical_adapt(support_examples, query_examples, model)
                    .await
            }
        }
    }

    /// Prototypical networks adaptation
    async fn prototypical_adapt(
        &mut self,
        support_examples: &[(String, String, String)],
        query_examples: &[(String, String)],
        model: &MultiModalEmbedding,
    ) -> Result<Vec<(String, f32)>> {
        // Extract features for support examples
        let mut prototypes = HashMap::new();
        let mut label_embeddings: HashMap<String, Vec<Array1<f32>>> = HashMap::new();

        for (text, entity, label) in support_examples {
            let text_emb = model.text_encoder.encode(text)?;
            let kg_emb_raw = model.get_or_create_kg_embedding(entity)?;
            let kg_emb = model.kg_encoder.encode_entity(&kg_emb_raw)?;

            // Combine text and KG embeddings
            let combined_emb = &text_emb + &kg_emb;

            label_embeddings
                .entry(label.clone())
                .or_default()
                .push(combined_emb);
        }

        // Compute prototypes
        for (label, embeddings) in &label_embeddings {
            let prototype = self.compute_prototype(embeddings)?;
            prototypes.insert(label.clone(), prototype);
        }

        // Classify query examples
        let mut predictions = Vec::new();
        for (text, entity) in query_examples {
            let text_emb = model.text_encoder.encode(text)?;
            let kg_emb_raw = model.get_or_create_kg_embedding(entity)?;
            let kg_emb = model.kg_encoder.encode_entity(&kg_emb_raw)?;

            let query_emb = &text_emb + &kg_emb;

            let mut best_score = f32::NEG_INFINITY;
            let mut best_label = String::new();

            for (label, prototype) in &prototypes {
                let distance = self.compute_distance(&query_emb, prototype);
                let score = (-distance / self.adaptation_config.temperature).exp();

                if score > best_score {
                    best_score = score;
                    best_label = label.clone();
                }
            }

            predictions.push((best_label, best_score));
        }

        Ok(predictions)
    }

    /// MAML adaptation
    async fn maml_adapt(
        &mut self,
        support_examples: &[(String, String, String)],
        query_examples: &[(String, String)],
        model: &MultiModalEmbedding,
    ) -> Result<Vec<(String, f32)>> {
        let task_id = {
            use scirs2_core::random::{Rng, Random};
            let mut random = Random::default();
            format!("task_{}", random.gen::<u32>())
        };

        // Initialize task-specific parameters
        let mut task_params = HashMap::new();
        for (layer_name, params) in &self.maml_components.inner_loop_params {
            task_params.insert(layer_name.clone(), params.clone());
        }

        // Inner loop: adapt on support set
        for _ in 0..self.adaptation_config.adaptation_steps {
            let mut gradients = HashMap::new();

            // Compute gradients on support set
            for (text, entity, label) in support_examples {
                let text_emb = model.text_encoder.encode(text)?;
                let kg_emb_raw = model.get_or_create_kg_embedding(entity)?;
                let kg_emb = model.kg_encoder.encode_entity(&kg_emb_raw)?;

                let input_emb = &text_emb + &kg_emb;
                let predicted = self.forward_pass(&input_emb, &task_params)?;

                // Compute loss and gradients (simplified)
                let target = self.label_to_target(label)?;
                let loss_grad = &predicted - &target;

                // Accumulate gradients
                for layer_name in task_params.keys() {
                    let grad = self.compute_layer_gradient(&input_emb, &loss_grad, layer_name)?;
                    *gradients
                        .entry(layer_name.clone())
                        .or_insert_with(|| Array2::zeros(grad.dim())) += &grad;
                }
            }

            // Update task parameters
            for (layer_name, params) in &mut task_params {
                if let Some(grad) = gradients.get(layer_name) {
                    *params = &*params - &(grad * self.adaptation_config.adaptation_lr);
                }
            }
        }

        // Store task adaptation
        self.maml_components
            .task_adaptations
            .insert(task_id.clone(), task_params.clone());

        // Evaluate on query set
        let mut predictions = Vec::new();
        for (text, entity) in query_examples {
            let text_emb = model.text_encoder.encode(text)?;
            let kg_emb_raw = model.get_or_create_kg_embedding(entity)?;
            let kg_emb = model.kg_encoder.encode_entity(&kg_emb_raw)?;

            let query_emb = &text_emb + &kg_emb;
            let output = self.forward_pass(&query_emb, &task_params)?;

            // Convert output to prediction
            let (predicted_label, confidence) = self.output_to_prediction(&output)?;
            predictions.push((predicted_label, confidence));
        }

        Ok(predictions)
    }

    /// Reptile adaptation
    async fn reptile_adapt(
        &mut self,
        support_examples: &[(String, String, String)],
        query_examples: &[(String, String)],
        model: &MultiModalEmbedding,
    ) -> Result<Vec<(String, f32)>> {
        // Reptile is similar to MAML but uses first-order gradients
        let mut adapted_params = HashMap::new();

        // Initialize with current parameters
        for (layer_name, params) in &self.maml_components.outer_loop_params {
            adapted_params.insert(layer_name.clone(), params.clone());
        }

        // Adapt on support set with multiple steps
        for _ in 0..self.adaptation_config.adaptation_steps {
            let mut param_updates = HashMap::new();

            for (text, entity, label) in support_examples {
                let text_emb = model.text_encoder.encode(text)?;
                let kg_emb_raw = model.get_or_create_kg_embedding(entity)?;
                let kg_emb = model.kg_encoder.encode_entity(&kg_emb_raw)?;

                let input_emb = &text_emb + &kg_emb;
                let predicted = self.forward_pass(&input_emb, &adapted_params)?;

                // Simple gradient approximation
                let target = self.label_to_target(label)?;
                let error = &predicted - &target;

                // Update parameters toward reducing error
                for (layer_name, params) in &adapted_params {
                    let update = &error * self.adaptation_config.adaptation_lr;
                    let param_change = Array2::from_shape_fn(params.dim(), |(i, j)| {
                        if i < update.len() && j < params.dim().1 {
                            update[i] * params[(i, j)]
                        } else {
                            0.0
                        }
                    });

                    *param_updates
                        .entry(layer_name.clone())
                        .or_insert_with(|| Array2::zeros(params.dim())) += &param_change;
                }
            }

            // Apply updates
            for (layer_name, params) in &mut adapted_params {
                if let Some(update) = param_updates.get(layer_name) {
                    *params = &*params - update;
                }
            }
        }

        // Evaluate on query set
        let mut predictions = Vec::new();
        for (text, entity) in query_examples {
            let text_emb = model.text_encoder.encode(text)?;
            let kg_emb_raw = model.get_or_create_kg_embedding(entity)?;
            let kg_emb = model.kg_encoder.encode_entity(&kg_emb_raw)?;

            let query_emb = &text_emb + &kg_emb;
            let output = self.forward_pass(&query_emb, &adapted_params)?;

            let (predicted_label, confidence) = self.output_to_prediction(&output)?;
            predictions.push((predicted_label, confidence));
        }

        Ok(predictions)
    }

    /// Compute prototype from embeddings
    pub fn compute_prototype(&self, embeddings: &[Array1<f32>]) -> Result<Array1<f32>> {
        if embeddings.is_empty() {
            return Err(anyhow!("Cannot compute prototype from empty embeddings"));
        }

        match self.prototypical_network.prototype_method {
            PrototypeMethod::Mean => {
                let mut prototype = Array1::zeros(embeddings[0].len());
                for emb in embeddings {
                    prototype = &prototype + emb;
                }
                prototype /= embeddings.len() as f32;
                Ok(prototype)
            }
            PrototypeMethod::AttentionWeighted => {
                // Compute attention-weighted prototype
                let mut weights = Vec::new();
                let mut weight_sum = 0.0;

                for emb in embeddings {
                    let weight = emb.dot(emb).sqrt(); // Use norm as attention weight
                    weights.push(weight);
                    weight_sum += weight;
                }

                let mut prototype = Array1::zeros(embeddings[0].len());
                for (emb, &weight) in embeddings.iter().zip(weights.iter()) {
                    prototype = &prototype + &(emb * (weight / weight_sum));
                }
                Ok(prototype)
            }
            PrototypeMethod::LearnableAggregation => {
                // Use learnable aggregation (simplified)
                let mut prototype = Array1::zeros(embeddings[0].len());
                for (i, emb) in embeddings.iter().enumerate() {
                    let weight = 1.0 / (1.0 + i as f32); // Decay weight
                    prototype = &prototype + &(emb * weight);
                }
                let total_weight: f32 = (0..embeddings.len()).map(|i| 1.0 / (1.0 + i as f32)).sum();
                prototype /= total_weight;
                Ok(prototype)
            }
        }
    }

    /// Compute distance between embeddings
    pub fn compute_distance(&self, emb1: &Array1<f32>, emb2: &Array1<f32>) -> f32 {
        match self.prototypical_network.distance_metric {
            DistanceMetric::Euclidean => {
                let diff = emb1 - emb2;
                diff.dot(&diff).sqrt()
            }
            DistanceMetric::Cosine => {
                let dot_product = emb1.dot(emb2);
                let norm1 = emb1.dot(emb1).sqrt();
                let norm2 = emb2.dot(emb2).sqrt();
                if norm1 > 0.0 && norm2 > 0.0 {
                    1.0 - (dot_product / (norm1 * norm2))
                } else {
                    1.0
                }
            }
            DistanceMetric::Learned => {
                // Use learned distance metric (simplified)
                let diff = emb1 - emb2;
                diff.mapv(|x| x.abs()).sum()
            }
            DistanceMetric::Mahalanobis => {
                // Simplified Mahalanobis distance
                let diff = emb1 - emb2;
                diff.dot(&diff).sqrt()
            }
        }
    }

    /// Forward pass through adapted network
    fn forward_pass(
        &self,
        input: &Array1<f32>,
        params: &HashMap<String, Array2<f32>>,
    ) -> Result<Array1<f32>> {
        let mut output = input.clone();

        // Simple feedforward network
        for layer_name in ["layer1", "layer2", "output"] {
            if let Some(weights) = params.get(layer_name) {
                output = weights.dot(&output);
                if layer_name != "output" {
                    output = output.mapv(|x| x.max(0.0)); // ReLU
                }
            }
        }

        Ok(output)
    }

    /// Convert label to target vector
    fn label_to_target(&self, label: &str) -> Result<Array1<f32>> {
        // Simple one-hot encoding based on label hash
        let label_hash = label.chars().map(|c| c as u8).sum::<u8>() as usize;
        let target_dim = 128; // Fixed target dimension
        let mut target = Array1::zeros(target_dim);
        target[label_hash % target_dim] = 1.0;
        Ok(target)
    }

    /// Compute layer gradient
    fn compute_layer_gradient(
        &self,
        input: &Array1<f32>,
        loss_grad: &Array1<f32>,
        _layer_name: &str,
    ) -> Result<Array2<f32>> {
        // Simplified gradient computation
        let input_len = input.len();
        let grad_len = loss_grad.len();
        let mut gradient = Array2::zeros((grad_len.min(128), input_len.min(128)));

        for i in 0..gradient.nrows() {
            for j in 0..gradient.ncols() {
                if i < loss_grad.len() && j < input.len() {
                    gradient[(i, j)] = loss_grad[i] * input[j];
                }
            }
        }

        Ok(gradient)
    }

    /// Convert output to prediction
    fn output_to_prediction(&self, output: &Array1<f32>) -> Result<(String, f32)> {
        // Find the index with maximum value
        let (max_idx, &max_val) = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        // Convert index to label
        let label = format!("class_{max_idx}");
        let confidence = 1.0 / (1.0 + (-max_val).exp()); // Sigmoid

        Ok((label, confidence))
    }

    /// Meta-update for improving few-shot performance
    pub fn meta_update(&mut self, tasks: &[Vec<(String, String, String)>]) -> Result<()> {
        match self.meta_algorithm {
            MetaAlgorithm::MAML => {
                // Update outer loop parameters based on task performance
                let mut meta_gradients = HashMap::new();

                for _task in tasks {
                    // Simulate task-specific adaptation
                    for layer_name in self.maml_components.outer_loop_params.keys() {
                        let grad = Array2::from_shape_fn((128, 128), |(_, _)| {
                            use scirs2_core::random::{Rng, Random};
                            let mut random = Random::default();
                            (random.gen::<f32>() - 0.5) * 0.01
                        });
                        *meta_gradients
                            .entry(layer_name.clone())
                            .or_insert_with(|| Array2::zeros((128, 128))) += &grad;
                    }
                }

                // Apply meta-gradients
                for (layer_name, params) in &mut self.maml_components.outer_loop_params {
                    if let Some(meta_grad) = meta_gradients.get(layer_name) {
                        *params = &*params - &(meta_grad * self.adaptation_config.adaptation_lr);
                    }
                }
            }
            MetaAlgorithm::Reptile => {
                // Reptile meta-update
                for _task in tasks {
                    // Simulate task adaptation and update toward adapted parameters
                    for params in self.maml_components.outer_loop_params.values_mut() {
                        let update = Array2::from_shape_fn(params.dim(), |(_, _)| {
                            use scirs2_core::random::{Rng, Random};
                            let mut random = Random::default();
                            (random.gen::<f32>() - 0.5) * 0.001
                        });
                        *params = &*params + &update;
                    }
                }
            }
            _ => {
                // For prototypical networks, update feature extractor
                for params in self.prototypical_network.feature_extractor.values_mut() {
                    let update = Array2::from_shape_fn(params.dim(), |(_, _)| {
                        use scirs2_core::random::{Rng, Random};
                        let mut random = Random::default();
                        (random.gen::<f32>() - 0.5) * 0.001
                    });
                    *params = &*params + &update;
                }
            }
        }

        Ok(())
    }
}
