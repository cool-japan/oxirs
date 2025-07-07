//! Advanced Context Management for OxiRS Chat
//!
//! Implements intelligent context management with sliding windows, topic tracking,
//! context summarization, and adaptive memory optimization.

use anyhow::{anyhow, Result};
use fastrand;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::{
    analytics::ConversationAnalytics, rag::QueryIntent, Message, MessageMetadata, MessageRole,
};

/// Configuration for context management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    pub sliding_window_size: usize,
    pub max_context_length: usize,
    pub enable_summarization: bool,
    pub summarization_threshold: usize,
    pub enable_topic_tracking: bool,
    pub topic_drift_threshold: f32,
    pub enable_importance_scoring: bool,
    pub memory_optimization_enabled: bool,
    pub adaptive_window_size: bool,
    pub context_compression_ratio: f32,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            sliding_window_size: 20,
            max_context_length: 4096,
            enable_summarization: true,
            summarization_threshold: 40,
            enable_topic_tracking: true,
            topic_drift_threshold: 0.7,
            enable_importance_scoring: true,
            memory_optimization_enabled: true,
            adaptive_window_size: true,
            context_compression_ratio: 0.6,
        }
    }
}

/// Neuromorphic Context Processing Engine
/// Implements brain-inspired processing patterns for context understanding
pub mod neuromorphic_context {
    use super::*;
    use std::collections::BTreeMap;

    /// Neuromorphic processing unit inspired by biological neurons
    #[derive(Debug, Clone)]
    pub struct NeuromorphicProcessor {
        pub dendrites: Vec<Dendrite>,
        pub soma: Soma,
        pub axon: Axon,
        pub synapses: Vec<Synapse>,
        pub activation_threshold: f64,
        pub refractory_period: Duration,
        pub learning_rate: f64,
    }

    impl NeuromorphicProcessor {
        pub fn new(input_size: usize) -> Self {
            let mut dendrites = Vec::new();
            for i in 0..input_size {
                dendrites.push(Dendrite::new(format!("dendrite_{}", i)));
            }

            Self {
                dendrites,
                soma: Soma::new(),
                axon: Axon::new(),
                synapses: Vec::new(),
                activation_threshold: 0.7,
                refractory_period: Duration::from_millis(100),
                learning_rate: 0.01,
            }
        }

        /// Process context input through neuromorphic pipeline
        pub fn process_context(&mut self, context_input: &ContextInput) -> NeuromorphicOutput {
            // Step 1: Dendritic processing
            let dendritic_signals = self.process_dendritic_input(context_input);

            // Step 2: Soma integration
            let soma_output = self.soma.integrate_signals(&dendritic_signals);

            // Step 3: Activation decision
            let activation = self.determine_activation(soma_output);

            // Step 4: Axonal transmission
            let axonal_output = if activation {
                self.axon.transmit_signal(soma_output)
            } else {
                AxonalOutput::default()
            };

            // Step 5: Synaptic learning
            self.update_synaptic_weights(&dendritic_signals, activation);

            NeuromorphicOutput {
                activation,
                signal_strength: soma_output,
                processed_features: self.extract_features(&dendritic_signals),
                learning_delta: self.calculate_learning_delta(&dendritic_signals, activation),
                attention_weights: self.calculate_attention_weights(&dendritic_signals),
            }
        }

        fn process_dendritic_input(&mut self, input: &ContextInput) -> Vec<DendriticSignal> {
            let mut signals = Vec::new();

            for (i, dendrite) in self.dendrites.iter_mut().enumerate() {
                let signal = dendrite.process_input(&input.features.get(i).unwrap_or(&0.0));
                signals.push(signal);
            }

            signals
        }

        fn determine_activation(&self, soma_output: f64) -> bool {
            soma_output > self.activation_threshold
        }

        fn update_synaptic_weights(&mut self, signals: &[DendriticSignal], activation: bool) {
            for (i, signal) in signals.iter().enumerate() {
                if i < self.synapses.len() {
                    let hebbian_update = if activation {
                        self.learning_rate * signal.strength
                    } else {
                        -self.learning_rate * signal.strength * 0.1
                    };

                    self.synapses[i].weight += hebbian_update;
                    self.synapses[i].weight = self.synapses[i].weight.clamp(-1.0, 1.0);
                }
            }
        }

        fn extract_features(&self, signals: &[DendriticSignal]) -> Vec<ExtractedFeature> {
            let mut features = Vec::new();

            // Pattern detection
            let pattern_strength = self.detect_patterns(signals);
            features.push(ExtractedFeature {
                name: "pattern_strength".to_string(),
                value: pattern_strength,
                confidence: 0.8,
            });

            // Temporal coherence
            let temporal_coherence = self.calculate_temporal_coherence(signals);
            features.push(ExtractedFeature {
                name: "temporal_coherence".to_string(),
                value: temporal_coherence,
                confidence: 0.7,
            });

            // Context stability
            let context_stability = self.assess_context_stability(signals);
            features.push(ExtractedFeature {
                name: "context_stability".to_string(),
                value: context_stability,
                confidence: 0.9,
            });

            features
        }

        fn detect_patterns(&self, signals: &[DendriticSignal]) -> f64 {
            if signals.len() < 2 {
                return 0.0;
            }

            let mut pattern_score = 0.0;
            let mut comparisons = 0;

            for i in 0..signals.len() - 1 {
                for j in i + 1..signals.len() {
                    let correlation = self.calculate_signal_correlation(&signals[i], &signals[j]);
                    pattern_score += correlation;
                    comparisons += 1;
                }
            }

            if comparisons > 0 {
                pattern_score / comparisons as f64
            } else {
                0.0
            }
        }

        fn calculate_signal_correlation(
            &self,
            signal1: &DendriticSignal,
            signal2: &DendriticSignal,
        ) -> f64 {
            let strength_correlation = 1.0 - (signal1.strength - signal2.strength).abs();
            let timing_correlation = 1.0 - (signal1.timing - signal2.timing).abs();

            (strength_correlation + timing_correlation) / 2.0
        }

        fn calculate_temporal_coherence(&self, signals: &[DendriticSignal]) -> f64 {
            if signals.is_empty() {
                return 0.0;
            }

            let mut coherence_sum = 0.0;
            let mut coherence_count = 0;

            for signal in signals {
                let expected_timing = signal.timing;
                let actual_timing = signal.timing;
                let coherence = 1.0 - (expected_timing - actual_timing).abs();
                coherence_sum += coherence;
                coherence_count += 1;
            }

            if coherence_count > 0 {
                coherence_sum / coherence_count as f64
            } else {
                0.0
            }
        }

        fn assess_context_stability(&self, signals: &[DendriticSignal]) -> f64 {
            if signals.is_empty() {
                return 0.0;
            }

            let mean_strength: f64 =
                signals.iter().map(|s| s.strength).sum::<f64>() / signals.len() as f64;
            let variance: f64 = signals
                .iter()
                .map(|s| (s.strength - mean_strength).powi(2))
                .sum::<f64>()
                / signals.len() as f64;

            1.0 / (1.0 + variance) // Higher stability = lower variance
        }

        fn calculate_learning_delta(&self, signals: &[DendriticSignal], activation: bool) -> f64 {
            let signal_strength_sum: f64 = signals.iter().map(|s| s.strength).sum();
            let learning_factor = if activation { 1.0 } else { -0.1 };

            self.learning_rate * signal_strength_sum * learning_factor
        }

        fn calculate_attention_weights(&self, signals: &[DendriticSignal]) -> Vec<f64> {
            let max_strength = signals.iter().map(|s| s.strength).fold(0.0, f64::max);

            if max_strength == 0.0 {
                return vec![1.0 / signals.len() as f64; signals.len()];
            }

            let mut weights = Vec::new();
            let mut weight_sum = 0.0;

            for signal in signals {
                let weight = (signal.strength / max_strength).exp();
                weights.push(weight);
                weight_sum += weight;
            }

            // Normalize weights
            for weight in &mut weights {
                *weight /= weight_sum;
            }

            weights
        }
    }

    #[derive(Debug, Clone)]
    pub struct Dendrite {
        pub id: String,
        pub receptive_field: Vec<f64>,
        pub activation_function: ActivationFunction,
        pub plasticity: f64,
    }

    impl Dendrite {
        pub fn new(id: String) -> Self {
            Self {
                id,
                receptive_field: vec![0.0; 10], // Default receptive field size
                activation_function: ActivationFunction::Sigmoid,
                plasticity: 0.1,
            }
        }

        pub fn process_input(&mut self, input: &f64) -> DendriticSignal {
            let signal_strength = self.activation_function.apply(*input);
            let timing = fastrand::f64(); // Simulated timing

            // Update receptive field with plasticity
            self.update_receptive_field(*input);

            DendriticSignal {
                strength: signal_strength,
                timing,
                source_id: self.id.clone(),
            }
        }

        fn update_receptive_field(&mut self, input: f64) {
            // Simple plasticity rule: strengthen connections for active inputs
            for field_value in &mut self.receptive_field {
                *field_value += self.plasticity * input * 0.01;
                *field_value = field_value.clamp(-1.0, 1.0);
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct Soma {
        pub membrane_potential: f64,
        pub integration_window: Duration,
        pub leak_conductance: f64,
    }

    impl Soma {
        pub fn new() -> Self {
            Self {
                membrane_potential: 0.0,
                integration_window: Duration::from_millis(50),
                leak_conductance: 0.1,
            }
        }

        pub fn integrate_signals(&mut self, signals: &[DendriticSignal]) -> f64 {
            // Reset membrane potential
            self.membrane_potential = 0.0;

            // Integrate incoming signals
            for signal in signals {
                self.membrane_potential += signal.strength;
            }

            // Apply leak conductance
            self.membrane_potential *= (1.0 - self.leak_conductance);

            self.membrane_potential
        }
    }

    #[derive(Debug, Clone)]
    pub struct Axon {
        pub conduction_velocity: f64,
        pub myelination: f64,
        pub branching_factor: usize,
    }

    impl Axon {
        pub fn new() -> Self {
            Self {
                conduction_velocity: 1.0,
                myelination: 0.8,
                branching_factor: 3,
            }
        }

        pub fn transmit_signal(&self, signal_strength: f64) -> AxonalOutput {
            let transmission_delay =
                Duration::from_millis(((1.0 - self.myelination) * 10.0) as u64);

            let amplified_signal = signal_strength * self.conduction_velocity;

            AxonalOutput {
                signal_strength: amplified_signal,
                transmission_delay,
                branching_outputs: vec![amplified_signal; self.branching_factor],
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct Synapse {
        pub weight: f64,
        pub neurotransmitter_type: NeurotransmitterType,
        pub plasticity_factor: f64,
        pub last_activation: Option<SystemTime>,
    }

    impl Synapse {
        pub fn new(neurotransmitter_type: NeurotransmitterType) -> Self {
            Self {
                weight: fastrand::f64() * 0.2 - 0.1,
                neurotransmitter_type,
                plasticity_factor: 0.1,
                last_activation: None,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub enum NeurotransmitterType {
        Excitatory, // Increases activation
        Inhibitory, // Decreases activation
        Modulatory, // Modifies learning
    }

    #[derive(Debug, Clone)]
    pub enum ActivationFunction {
        Sigmoid,
        ReLU,
        Tanh,
        Leaky,
    }

    impl ActivationFunction {
        pub fn apply(&self, input: f64) -> f64 {
            match self {
                ActivationFunction::Sigmoid => 1.0 / (1.0 + (-input).exp()),
                ActivationFunction::ReLU => input.max(0.0),
                ActivationFunction::Tanh => input.tanh(),
                ActivationFunction::Leaky => {
                    if input > 0.0 {
                        input
                    } else {
                        0.01 * input
                    }
                }
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct DendriticSignal {
        pub strength: f64,
        pub timing: f64,
        pub source_id: String,
    }

    #[derive(Debug, Clone, Default)]
    pub struct AxonalOutput {
        pub signal_strength: f64,
        pub transmission_delay: Duration,
        pub branching_outputs: Vec<f64>,
    }

    #[derive(Debug, Clone)]
    pub struct ContextInput {
        pub features: Vec<f64>,
        pub metadata: HashMap<String, String>,
        pub timestamp: SystemTime,
    }

    #[derive(Debug, Clone)]
    pub struct NeuromorphicOutput {
        pub activation: bool,
        pub signal_strength: f64,
        pub processed_features: Vec<ExtractedFeature>,
        pub learning_delta: f64,
        pub attention_weights: Vec<f64>,
    }

    #[derive(Debug, Clone)]
    pub struct ExtractedFeature {
        pub name: String,
        pub value: f64,
        pub confidence: f64,
    }

    /// Neuromorphic Context Manager that integrates with existing context system
    pub struct NeuromorphicContextManager {
        processors: Vec<NeuromorphicProcessor>,
        context_memory: ContextMemory,
        attention_mechanism: AttentionMechanism,
        learning_system: LearningSystem,
    }

    impl NeuromorphicContextManager {
        pub fn new(config: &ContextConfig) -> Self {
            let mut processors = Vec::new();

            // Create specialized processors for different context aspects
            processors.push(NeuromorphicProcessor::new(10)); // Semantic processor
            processors.push(NeuromorphicProcessor::new(8)); // Temporal processor
            processors.push(NeuromorphicProcessor::new(6)); // Emotional processor
            processors.push(NeuromorphicProcessor::new(12)); // Contextual processor

            Self {
                processors,
                context_memory: ContextMemory::new(config.max_context_length),
                attention_mechanism: AttentionMechanism::new(),
                learning_system: LearningSystem::new(config.adaptive_window_size),
            }
        }

        /// Process context through neuromorphic pipeline
        pub fn process_neuromorphic_context(
            &mut self,
            messages: &[Message],
        ) -> NeuromorphicContextResult {
            let mut processor_outputs = Vec::new();

            // Extract features from messages
            let context_features = self.extract_context_features(messages);

            // Prepare processor inputs before mutable iteration
            let processor_inputs: Vec<_> = (0..self.processors.len())
                .map(|i| self.prepare_processor_input(&context_features, i))
                .collect();

            // Process through each specialized processor
            for (i, processor) in self.processors.iter_mut().enumerate() {
                let output = processor.process_context(&processor_inputs[i]);
                processor_outputs.push(output);
            }

            // Integrate outputs through attention mechanism
            let integrated_output = self
                .attention_mechanism
                .integrate_outputs(&processor_outputs);

            // Update context memory
            self.context_memory.update(&integrated_output, messages);

            // Learn from processing results
            self.learning_system
                .learn_from_context(&integrated_output, messages);

            NeuromorphicContextResult {
                enhanced_context: integrated_output,
                attention_map: self.attention_mechanism.get_attention_map(),
                learning_insights: self.learning_system.get_insights(),
                memory_state: self.context_memory.get_state(),
            }
        }

        fn extract_context_features(&self, messages: &[Message]) -> ContextFeatures {
            let mut features = ContextFeatures::new();

            for message in messages {
                // Semantic features
                features
                    .semantic_features
                    .push(self.extract_semantic_features(message));

                // Temporal features
                features
                    .temporal_features
                    .push(self.extract_temporal_features(message));

                // Emotional features
                features
                    .emotional_features
                    .push(self.extract_emotional_features(message));

                // Contextual features
                features
                    .contextual_features
                    .push(self.extract_contextual_features(message));
            }

            features
        }

        fn extract_semantic_features(&self, message: &Message) -> Vec<f64> {
            let content = message.content.to_text();
            let word_count = content.split_whitespace().count() as f64;
            let char_count = content.len() as f64;
            let avg_word_length = if word_count > 0.0 {
                char_count / word_count
            } else {
                0.0
            };

            vec![
                word_count / 100.0,                      // Normalized word count
                char_count / 1000.0,                     // Normalized character count
                avg_word_length / 10.0,                  // Normalized average word length
                self.calculate_complexity(content),      // Content complexity
                self.calculate_informativeness(content), // Information density
            ]
        }

        fn extract_temporal_features(&self, message: &Message) -> Vec<f64> {
            let now = SystemTime::now();
            let message_age = now
                .duration_since(message.timestamp.into())
                .unwrap_or_default()
                .as_secs() as f64;

            vec![
                message_age / 3600.0,                      // Age in hours
                (message_age % 86400.0) / 86400.0,         // Time of day factor
                self.calculate_recency_score(message_age), // Recency importance
            ]
        }

        fn extract_emotional_features(&self, message: &Message) -> Vec<f64> {
            let content = message.content.to_text();

            vec![
                self.detect_emotional_valence(content),
                self.detect_emotional_arousal(content),
                self.detect_emotional_dominance(content),
            ]
        }

        fn extract_contextual_features(&self, message: &Message) -> Vec<f64> {
            vec![
                if message.parent_message_id.is_some() {
                    1.0
                } else {
                    0.0
                }, // Is reply
                message.reactions.len() as f64 / 10.0, // Reaction count
                message.attachments.len() as f64 / 5.0, // Attachment count
                if message.thread_id.is_some() {
                    1.0
                } else {
                    0.0
                }, // Is threaded
            ]
        }

        fn calculate_complexity(&self, content: &str) -> f64 {
            let unique_words = content.split_whitespace().collect::<HashSet<_>>().len();
            let total_words = content.split_whitespace().count();

            if total_words > 0 {
                unique_words as f64 / total_words as f64
            } else {
                0.0
            }
        }

        fn calculate_informativeness(&self, content: &str) -> f64 {
            let information_words = ["what", "how", "why", "when", "where", "who", "which"];
            let content_lower = content.to_lowercase();

            let info_word_count = information_words
                .iter()
                .filter(|word| content_lower.contains(*word))
                .count();

            (info_word_count as f64 / information_words.len() as f64).min(1.0)
        }

        fn calculate_recency_score(&self, age_seconds: f64) -> f64 {
            // Exponential decay function for recency
            (-age_seconds / 3600.0).exp() // Decay over hours
        }

        fn detect_emotional_valence(&self, content: &str) -> f64 {
            let positive_words = [
                "good",
                "great",
                "excellent",
                "amazing",
                "wonderful",
                "fantastic",
            ];
            let negative_words = [
                "bad",
                "terrible",
                "awful",
                "horrible",
                "disappointing",
                "frustrating",
            ];

            let content_lower = content.to_lowercase();
            let positive_count = positive_words
                .iter()
                .filter(|word| content_lower.contains(*word))
                .count();
            let negative_count = negative_words
                .iter()
                .filter(|word| content_lower.contains(*word))
                .count();

            let total_count = positive_count + negative_count;
            if total_count == 0 {
                0.0 // Neutral
            } else {
                (positive_count as f64 - negative_count as f64) / total_count as f64
            }
        }

        fn detect_emotional_arousal(&self, content: &str) -> f64 {
            let high_arousal_words = [
                "excited", "angry", "thrilled", "furious", "ecstatic", "outraged",
            ];
            let content_lower = content.to_lowercase();

            let arousal_count = high_arousal_words
                .iter()
                .filter(|word| content_lower.contains(*word))
                .count();

            (arousal_count as f64 / high_arousal_words.len() as f64).min(1.0)
        }

        fn detect_emotional_dominance(&self, content: &str) -> f64 {
            let dominant_words = ["command", "control", "decide", "lead", "manage", "direct"];
            let content_lower = content.to_lowercase();

            let dominance_count = dominant_words
                .iter()
                .filter(|word| content_lower.contains(*word))
                .count();

            (dominance_count as f64 / dominant_words.len() as f64).min(1.0)
        }

        fn prepare_processor_input(
            &self,
            features: &ContextFeatures,
            processor_index: usize,
        ) -> ContextInput {
            let selected_features = match processor_index {
                0 => features.get_semantic_features(),   // Semantic processor
                1 => features.get_temporal_features(),   // Temporal processor
                2 => features.get_emotional_features(),  // Emotional processor
                3 => features.get_contextual_features(), // Contextual processor
                _ => features.get_combined_features(),   // Default
            };

            ContextInput {
                features: selected_features,
                metadata: HashMap::new(),
                timestamp: SystemTime::now(),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct ContextFeatures {
        pub semantic_features: Vec<Vec<f64>>,
        pub temporal_features: Vec<Vec<f64>>,
        pub emotional_features: Vec<Vec<f64>>,
        pub contextual_features: Vec<Vec<f64>>,
    }

    impl ContextFeatures {
        pub fn new() -> Self {
            Self {
                semantic_features: Vec::new(),
                temporal_features: Vec::new(),
                emotional_features: Vec::new(),
                contextual_features: Vec::new(),
            }
        }

        pub fn get_semantic_features(&self) -> Vec<f64> {
            self.semantic_features.iter().flatten().cloned().collect()
        }

        pub fn get_temporal_features(&self) -> Vec<f64> {
            self.temporal_features.iter().flatten().cloned().collect()
        }

        pub fn get_emotional_features(&self) -> Vec<f64> {
            self.emotional_features.iter().flatten().cloned().collect()
        }

        pub fn get_contextual_features(&self) -> Vec<f64> {
            self.contextual_features.iter().flatten().cloned().collect()
        }

        pub fn get_combined_features(&self) -> Vec<f64> {
            let mut combined = Vec::new();
            combined.extend(self.get_semantic_features());
            combined.extend(self.get_temporal_features());
            combined.extend(self.get_emotional_features());
            combined.extend(self.get_contextual_features());
            combined
        }
    }

    #[derive(Debug, Clone)]
    pub struct ContextMemory {
        pub capacity: usize,
        pub stored_contexts: VecDeque<StoredContext>,
        pub importance_threshold: f64,
    }

    impl ContextMemory {
        pub fn new(capacity: usize) -> Self {
            Self {
                capacity,
                stored_contexts: VecDeque::new(),
                importance_threshold: 0.5,
            }
        }

        pub fn update(&mut self, output: &NeuromorphicOutput, messages: &[Message]) {
            let context = StoredContext {
                id: Uuid::new_v4().to_string(),
                processed_features: output.processed_features.clone(),
                importance_score: output.signal_strength,
                timestamp: SystemTime::now(),
                message_count: messages.len(),
            };

            if context.importance_score > self.importance_threshold {
                self.stored_contexts.push_back(context);

                // Maintain capacity
                while self.stored_contexts.len() > self.capacity {
                    self.stored_contexts.pop_front();
                }
            }
        }

        pub fn get_state(&self) -> MemoryState {
            MemoryState {
                stored_count: self.stored_contexts.len(),
                capacity_utilization: self.stored_contexts.len() as f64 / self.capacity as f64,
                average_importance: self.calculate_average_importance(),
                oldest_context_age: self.get_oldest_context_age(),
            }
        }

        fn calculate_average_importance(&self) -> f64 {
            if self.stored_contexts.is_empty() {
                return 0.0;
            }

            let total_importance: f64 = self
                .stored_contexts
                .iter()
                .map(|ctx| ctx.importance_score)
                .sum();

            total_importance / self.stored_contexts.len() as f64
        }

        fn get_oldest_context_age(&self) -> Duration {
            self.stored_contexts
                .front()
                .map(|ctx| {
                    SystemTime::now()
                        .duration_since(ctx.timestamp)
                        .unwrap_or_default()
                })
                .unwrap_or_default()
        }
    }

    #[derive(Debug, Clone)]
    pub struct StoredContext {
        pub id: String,
        pub processed_features: Vec<ExtractedFeature>,
        pub importance_score: f64,
        pub timestamp: SystemTime,
        pub message_count: usize,
    }

    #[derive(Debug, Clone)]
    pub struct AttentionMechanism {
        pub attention_weights: Vec<f64>,
        pub focus_threshold: f64,
        pub attention_decay: f64,
    }

    impl AttentionMechanism {
        pub fn new() -> Self {
            Self {
                attention_weights: Vec::new(),
                focus_threshold: 0.3,
                attention_decay: 0.95,
            }
        }

        pub fn integrate_outputs(&mut self, outputs: &[NeuromorphicOutput]) -> NeuromorphicOutput {
            if outputs.is_empty() {
                return NeuromorphicOutput {
                    activation: false,
                    signal_strength: 0.0,
                    processed_features: Vec::new(),
                    learning_delta: 0.0,
                    attention_weights: Vec::new(),
                };
            }

            // Calculate attention weights based on signal strength
            let total_strength: f64 = outputs.iter().map(|o| o.signal_strength).sum();
            let mut new_attention_weights = Vec::new();

            for output in outputs {
                let weight = if total_strength > 0.0 {
                    output.signal_strength / total_strength
                } else {
                    1.0 / outputs.len() as f64
                };
                new_attention_weights.push(weight);
            }

            self.attention_weights = new_attention_weights.clone();

            // Weighted integration of features
            let mut integrated_features = Vec::new();
            let mut integrated_signal_strength = 0.0;
            let mut integrated_learning_delta = 0.0;
            let mut activation_count = 0;

            for (i, output) in outputs.iter().enumerate() {
                let weight = new_attention_weights[i];

                integrated_signal_strength += output.signal_strength * weight;
                integrated_learning_delta += output.learning_delta * weight;

                if output.activation {
                    activation_count += 1;
                }

                // Weight the features
                for feature in &output.processed_features {
                    integrated_features.push(ExtractedFeature {
                        name: format!("weighted_{}", feature.name),
                        value: feature.value * weight,
                        confidence: feature.confidence * weight,
                    });
                }
            }

            let integrated_activation = activation_count > outputs.len() / 2;

            NeuromorphicOutput {
                activation: integrated_activation,
                signal_strength: integrated_signal_strength,
                processed_features: integrated_features,
                learning_delta: integrated_learning_delta,
                attention_weights: new_attention_weights,
            }
        }

        pub fn get_attention_map(&self) -> AttentionMap {
            AttentionMap {
                weights: self.attention_weights.clone(),
                focus_areas: self.identify_focus_areas(),
                attention_distribution: self.calculate_attention_distribution(),
            }
        }

        fn identify_focus_areas(&self) -> Vec<String> {
            let mut focus_areas = Vec::new();

            for (i, weight) in self.attention_weights.iter().enumerate() {
                if *weight > self.focus_threshold {
                    focus_areas.push(match i {
                        0 => "Semantic Processing".to_string(),
                        1 => "Temporal Processing".to_string(),
                        2 => "Emotional Processing".to_string(),
                        3 => "Contextual Processing".to_string(),
                        _ => format!("Processor {}", i),
                    });
                }
            }

            focus_areas
        }

        fn calculate_attention_distribution(&self) -> AttentionDistribution {
            let max_attention = self
                .attention_weights
                .iter()
                .fold(0.0_f64, |a, b| a.max(*b));
            let min_attention = self
                .attention_weights
                .iter()
                .fold(1.0_f64, |a, b| a.min(*b));
            let mean_attention = if !self.attention_weights.is_empty() {
                self.attention_weights.iter().sum::<f64>() / self.attention_weights.len() as f64
            } else {
                0.0
            };

            AttentionDistribution {
                max_attention,
                min_attention,
                mean_attention,
                attention_spread: max_attention - min_attention,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct LearningSystem {
        pub learning_history: Vec<LearningEvent>,
        pub adaptation_rate: f64,
        pub performance_metrics: PerformanceMetrics,
    }

    impl LearningSystem {
        pub fn new(adaptive: bool) -> Self {
            Self {
                learning_history: Vec::new(),
                adaptation_rate: if adaptive { 0.1 } else { 0.01 },
                performance_metrics: PerformanceMetrics::new(),
            }
        }

        pub fn learn_from_context(&mut self, output: &NeuromorphicOutput, messages: &[Message]) {
            let learning_event = LearningEvent {
                id: Uuid::new_v4().to_string(),
                timestamp: SystemTime::now(),
                learning_delta: output.learning_delta,
                context_complexity: self.assess_context_complexity(messages),
                performance_improvement: self.calculate_performance_improvement(output),
            };

            self.learning_history.push(learning_event);
            self.update_performance_metrics(output);

            // Maintain learning history size
            if self.learning_history.len() > 1000 {
                self.learning_history.remove(0);
            }
        }

        pub fn get_insights(&self) -> LearningInsights {
            LearningInsights {
                total_learning_events: self.learning_history.len(),
                average_learning_delta: self.calculate_average_learning_delta(),
                learning_trend: self.calculate_learning_trend(),
                adaptation_effectiveness: self.assess_adaptation_effectiveness(),
                performance_summary: self.performance_metrics.clone(),
            }
        }

        fn assess_context_complexity(&self, messages: &[Message]) -> f64 {
            let total_length: usize = messages.iter().map(|m| m.content.len()).sum();
            let unique_roles = messages
                .iter()
                .map(|m| &m.role)
                .collect::<HashSet<_>>()
                .len();
            let thread_complexity = messages.iter().filter(|m| m.thread_id.is_some()).count();

            ((total_length as f64 / 1000.0)
                + (unique_roles as f64 * 0.5)
                + (thread_complexity as f64 * 0.3))
                .min(10.0)
        }

        fn calculate_performance_improvement(&self, output: &NeuromorphicOutput) -> f64 {
            if self.learning_history.is_empty() {
                return 0.0;
            }

            let recent_performance = self
                .learning_history
                .iter()
                .rev()
                .take(10)
                .map(|event| event.learning_delta.abs())
                .sum::<f64>()
                / 10.0;

            let historical_performance = self
                .learning_history
                .iter()
                .take(self.learning_history.len().saturating_sub(10))
                .map(|event| event.learning_delta.abs())
                .sum::<f64>()
                / (self.learning_history.len() - 10).max(1) as f64;

            recent_performance - historical_performance
        }

        fn update_performance_metrics(&mut self, output: &NeuromorphicOutput) {
            self.performance_metrics.total_processed += 1;
            self.performance_metrics.total_learning_delta += output.learning_delta.abs();

            if output.activation {
                self.performance_metrics.activation_count += 1;
            }

            self.performance_metrics.average_signal_strength =
                (self.performance_metrics.average_signal_strength
                    * (self.performance_metrics.total_processed - 1) as f64
                    + output.signal_strength)
                    / self.performance_metrics.total_processed as f64;
        }

        fn calculate_average_learning_delta(&self) -> f64 {
            if self.learning_history.is_empty() {
                return 0.0;
            }

            self.learning_history
                .iter()
                .map(|event| event.learning_delta)
                .sum::<f64>()
                / self.learning_history.len() as f64
        }

        fn calculate_learning_trend(&self) -> f64 {
            if self.learning_history.len() < 2 {
                return 0.0;
            }

            let recent_half = self.learning_history.len() / 2;
            let recent_avg = self
                .learning_history
                .iter()
                .skip(recent_half)
                .map(|event| event.learning_delta)
                .sum::<f64>()
                / (self.learning_history.len() - recent_half) as f64;

            let early_avg = self
                .learning_history
                .iter()
                .take(recent_half)
                .map(|event| event.learning_delta)
                .sum::<f64>()
                / recent_half as f64;

            recent_avg - early_avg
        }

        fn assess_adaptation_effectiveness(&self) -> f64 {
            let performance_variance = self.calculate_performance_variance();
            let learning_consistency = self.calculate_learning_consistency();

            (1.0 / (1.0 + performance_variance)) * learning_consistency
        }

        fn calculate_performance_variance(&self) -> f64 {
            if self.learning_history.len() < 2 {
                return 0.0;
            }

            let mean = self.calculate_average_learning_delta();
            let variance = self
                .learning_history
                .iter()
                .map(|event| (event.learning_delta - mean).powi(2))
                .sum::<f64>()
                / self.learning_history.len() as f64;

            variance.sqrt()
        }

        fn calculate_learning_consistency(&self) -> f64 {
            if self.learning_history.len() < 3 {
                return 1.0;
            }

            let mut consistency_score = 0.0;
            let mut comparisons = 0;

            for i in 1..self.learning_history.len() {
                let current_delta = self.learning_history[i].learning_delta;
                let previous_delta = self.learning_history[i - 1].learning_delta;

                let consistency = 1.0 - (current_delta - previous_delta).abs();
                consistency_score += consistency;
                comparisons += 1;
            }

            if comparisons > 0 {
                consistency_score / comparisons as f64
            } else {
                1.0
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct LearningEvent {
        pub id: String,
        pub timestamp: SystemTime,
        pub learning_delta: f64,
        pub context_complexity: f64,
        pub performance_improvement: f64,
    }

    #[derive(Debug, Clone)]
    pub struct PerformanceMetrics {
        pub total_processed: usize,
        pub activation_count: usize,
        pub total_learning_delta: f64,
        pub average_signal_strength: f64,
    }

    impl PerformanceMetrics {
        pub fn new() -> Self {
            Self {
                total_processed: 0,
                activation_count: 0,
                total_learning_delta: 0.0,
                average_signal_strength: 0.0,
            }
        }
    }

    // Additional supporting structures
    #[derive(Debug, Clone)]
    pub struct NeuromorphicContextResult {
        pub enhanced_context: NeuromorphicOutput,
        pub attention_map: AttentionMap,
        pub learning_insights: LearningInsights,
        pub memory_state: MemoryState,
    }

    #[derive(Debug, Clone)]
    pub struct AttentionMap {
        pub weights: Vec<f64>,
        pub focus_areas: Vec<String>,
        pub attention_distribution: AttentionDistribution,
    }

    #[derive(Debug, Clone)]
    pub struct AttentionDistribution {
        pub max_attention: f64,
        pub min_attention: f64,
        pub mean_attention: f64,
        pub attention_spread: f64,
    }

    #[derive(Debug, Clone)]
    pub struct LearningInsights {
        pub total_learning_events: usize,
        pub average_learning_delta: f64,
        pub learning_trend: f64,
        pub adaptation_effectiveness: f64,
        pub performance_summary: PerformanceMetrics,
    }

    #[derive(Debug, Clone)]
    pub struct MemoryState {
        pub stored_count: usize,
        pub capacity_utilization: f64,
        pub average_importance: f64,
        pub oldest_context_age: Duration,
    }
}

// Supporting component definitions
struct TopicTracker {
    _config: ContextConfig,
}

impl TopicTracker {
    fn new(_config: &ContextConfig) -> Self {
        Self {
            _config: _config.clone(),
        }
    }

    async fn process_message(&mut self, _message: &Message) -> Result<TopicUpdate> {
        Ok(TopicUpdate {
            new_topics: Vec::new(),
            topic_changes: Vec::new(),
            drift_detected: false,
        })
    }

    async fn get_current_topics(&self) -> Vec<Topic> {
        Vec::new()
    }

    async fn get_current_topic(&self) -> Option<String> {
        let topics = self.get_current_topics().await;
        topics.first().map(|topic| topic.name.clone())
    }

    async fn transition_to_topic(
        &mut self,
        topic: &str,
        _hint: Option<&str>,
    ) -> Result<TopicTransition> {
        Ok(TopicTransition {
            from_topic: None,
            to_topic: topic.to_string(),
            transition_reason: "User initiated".to_string(),
            confidence: 0.8,
            timestamp: SystemTime::now(),
        })
    }

    async fn topic_count(&self) -> usize {
        0
    }
}

struct ImportanceScorer {
    _config: ContextConfig,
}

impl ImportanceScorer {
    fn new(_config: &ContextConfig) -> Self {
        Self {
            _config: _config.clone(),
        }
    }

    async fn score_message(&self, _message: &Message) -> f32 {
        0.5
    }

    async fn update_for_context_switch(&mut self, _transition: &TopicTransition) -> Result<()> {
        Ok(())
    }

    async fn average_score(&self) -> f32 {
        0.5
    }
}

struct SummarizationEngine {
    _config: ContextConfig,
}

impl SummarizationEngine {
    fn new(_config: &ContextConfig) -> Self {
        Self {
            _config: _config.clone(),
        }
    }

    async fn summarize_messages(&self, _messages: &[Message]) -> Result<ContextSummary> {
        Ok(ContextSummary {
            text: "Summary placeholder".to_string(),
            key_points: vec!["Summary placeholder".to_string()],
            entities_mentioned: vec![],
            topics_covered: vec![],
            created_at: SystemTime::now(),
        })
    }

    async fn summarization_count(&self) -> usize {
        0
    }
}

struct MemoryOptimizer {
    _config: ContextConfig,
}

impl MemoryOptimizer {
    fn new(_config: &ContextConfig) -> Self {
        Self {
            _config: _config.clone(),
        }
    }

    async fn optimize_context(
        &mut self,
        _window: &mut ContextWindow,
    ) -> Result<OptimizationUpdate> {
        Ok(OptimizationUpdate {
            memory_saved: 0,
            operations_performed: vec![],
            efficiency_improvement: 0.0,
        })
    }

    async fn optimization_count(&self) -> usize {
        0
    }
}

/// Advanced context manager
pub struct AdvancedContextManager {
    config: ContextConfig,
    context_window: ContextWindow,
    topic_tracker: TopicTracker,
    importance_scorer: ImportanceScorer,
    summarization_engine: SummarizationEngine,
    memory_optimizer: MemoryOptimizer,
}

impl AdvancedContextManager {
    pub fn new(config: ContextConfig) -> Self {
        Self {
            context_window: ContextWindow::new(&config),
            topic_tracker: TopicTracker::new(&config),
            importance_scorer: ImportanceScorer::new(&config),
            summarization_engine: SummarizationEngine::new(&config),
            memory_optimizer: MemoryOptimizer::new(&config),
            config,
        }
    }

    /// Process a new message and update context
    pub async fn process_message(
        &mut self,
        message: &Message,
        conversation_analytics: Option<&ConversationAnalytics>,
    ) -> Result<ContextUpdate> {
        let start_time = SystemTime::now();

        // Calculate importance score
        let importance_score = self.importance_scorer.score_message(message).await;

        // Update context window
        let window_update = self
            .context_window
            .add_message(message.clone(), importance_score)
            .await?;

        // Track topic changes
        let topic_update = if self.config.enable_topic_tracking {
            Some(self.topic_tracker.process_message(message).await?)
        } else {
            None
        };

        // Check if summarization is needed
        let summarization_update =
            if self.config.enable_summarization && self.context_window.should_summarize().await {
                Some(self.perform_summarization().await?)
            } else {
                None
            };

        // Optimize memory if needed
        let optimization_update = if self.config.memory_optimization_enabled {
            Some(
                self.memory_optimizer
                    .optimize_context(&mut self.context_window)
                    .await?,
            )
        } else {
            None
        };

        let processing_time = start_time.elapsed().unwrap_or(Duration::ZERO);

        Ok(ContextUpdate {
            message_processed: message.id.clone(),
            importance_score,
            window_update,
            topic_update,
            summarization_update,
            optimization_update,
            processing_time,
        })
    }

    /// Get current context for LLM
    pub async fn get_current_context(&self) -> Result<AssembledContext> {
        let effective_messages = self.context_window.get_effective_messages().await?;
        let current_topics = self.topic_tracker.get_current_topics().await;
        let context_summary = self.context_window.get_summary().await;

        // Assemble context with proper ordering and formatting
        let mut context_text = String::new();

        // Add summary if available
        if let Some(summary) = &context_summary {
            context_text.push_str("## Conversation Summary\n");
            context_text.push_str(summary);
            context_text.push_str("\n\n");
        }

        // Add current topics
        if !current_topics.is_empty() {
            context_text.push_str("## Current Topics\n");
            for topic in &current_topics {
                context_text.push_str(&format!(
                    "- {} (confidence: {:.2})\n",
                    topic.name, topic.confidence
                ));
            }
            context_text.push_str("\n");
        }

        // Add recent messages
        context_text.push_str("## Recent Messages\n");
        for message in &effective_messages {
            let role_indicator = match message.role {
                MessageRole::User => "User",
                MessageRole::Assistant => "Assistant",
                MessageRole::System => "System",
                MessageRole::Function => "Function",
            };
            context_text.push_str(&format!("{}: {}\n", role_indicator, message.content));
        }

        // Calculate quality metrics
        let quality_score = self
            .calculate_context_quality(&effective_messages, &current_topics)
            .await;
        let coverage_score = self.calculate_coverage_score(&effective_messages).await;

        // Calculate values before moving into the struct
        let token_count = self.estimate_token_count(&context_text).await;
        let structured_context = self.extract_structured_context(&effective_messages).await?;

        Ok(AssembledContext {
            context_text,
            effective_messages,
            current_topics,
            context_summary,
            quality_score,
            coverage_score,
            token_count,
            structured_context,
        })
    }

    /// Handle context switching
    pub async fn switch_context(
        &mut self,
        new_topic: &str,
        context_hint: Option<&str>,
    ) -> Result<ContextSwitch> {
        info!("Switching context to topic: {}", new_topic);

        // Save current context state
        let current_topic = self.topic_tracker.get_current_topic().await;
        let previous_state = self
            .context_window
            .get_state_snapshot_with_topic(current_topic)
            .await;

        // Perform topic transition
        let topic_transition = self
            .topic_tracker
            .transition_to_topic(new_topic, context_hint)
            .await?;

        // Adjust context window for new topic
        let window_adjustment = self
            .context_window
            .adjust_for_topic(&topic_transition)
            .await?;

        // Update importance scoring for new context
        self.importance_scorer
            .update_for_context_switch(&topic_transition)
            .await?;

        // Implement actual preservation logic
        let context_preserved = self
            .evaluate_context_preservation(&previous_state, &topic_transition, &window_adjustment)
            .await?;

        Ok(ContextSwitch {
            previous_state,
            new_topic: new_topic.to_string(),
            topic_transition,
            window_adjustment,
            context_preserved,
        })
    }

    /// Pin an important message
    pub async fn pin_message(&mut self, message_id: &str, reason: PinReason) -> Result<()> {
        self.context_window.pin_message(message_id, reason).await
    }

    /// Unpin a message
    pub async fn unpin_message(&mut self, message_id: &str) -> Result<()> {
        self.context_window.unpin_message(message_id).await
    }

    /// Evaluate whether context was properly preserved during a topic switch
    async fn evaluate_context_preservation(
        &self,
        previous_state: &ContextState,
        topic_transition: &TopicTransition,
        window_adjustment: &WindowAdjustment,
    ) -> Result<bool> {
        let mut preservation_score = 0.0;
        let mut factors_checked = 0;

        // Factor 1: Topic transition confidence (weight: 0.3)
        if topic_transition.confidence >= 0.8 {
            preservation_score += 0.3;
        } else if topic_transition.confidence >= 0.6 {
            preservation_score += 0.15;
        }
        factors_checked += 1;

        // Factor 2: Message retention (weight: 0.25)
        let current_state = self.context_window.get_state_snapshot().await;
        let message_retention_ratio = if previous_state.message_count > 0 {
            current_state.message_count as f32 / previous_state.message_count as f32
        } else {
            1.0
        };

        if message_retention_ratio >= 0.8 {
            preservation_score += 0.25;
        } else if message_retention_ratio >= 0.6 {
            preservation_score += 0.15;
        } else if message_retention_ratio >= 0.4 {
            preservation_score += 0.1;
        }
        factors_checked += 1;

        // Factor 3: Pinned messages preservation (weight: 0.2)
        if previous_state.pinned_count > 0 {
            let pinned_retention_ratio =
                current_state.pinned_count as f32 / previous_state.pinned_count as f32;
            if pinned_retention_ratio >= 0.9 {
                preservation_score += 0.2;
            } else if pinned_retention_ratio >= 0.7 {
                preservation_score += 0.1;
            }
        } else {
            // If there were no pinned messages, this factor doesn't penalize
            preservation_score += 0.2;
        }
        factors_checked += 1;

        // Factor 4: Context continuity (weight: 0.15)
        if previous_state.has_summary && current_state.has_summary {
            preservation_score += 0.15;
        } else if !previous_state.has_summary && !current_state.has_summary {
            preservation_score += 0.1; // Consistency bonus
        } else if !previous_state.has_summary && current_state.has_summary {
            preservation_score += 0.05; // Slight bonus for improvement
        }
        factors_checked += 1;

        // Factor 5: Window adjustment success (weight: 0.1)
        let adjustment_success_score = [
            window_adjustment.messages_reordered,
            window_adjustment.importance_rescored,
            window_adjustment.window_size_adjusted,
        ]
        .iter()
        .filter(|&&success| success)
        .count() as f32
            / 3.0;

        preservation_score += 0.1 * adjustment_success_score;
        factors_checked += 1;

        // Calculate final score and determine if context was preserved
        let final_score = preservation_score;
        let context_preserved = final_score >= 0.7; // Require 70% preservation score

        debug!(
            "Context preservation evaluation: score={:.2}, factors_checked={}, preserved={}",
            final_score, factors_checked, context_preserved
        );

        if !context_preserved {
            warn!(
                "Context preservation failed: transition_confidence={:.2}, message_retention={:.2}, final_score={:.2}",
                topic_transition.confidence,
                message_retention_ratio,
                final_score
            );
        } else {
            info!(
                "Context successfully preserved: score={:.2}, transition to topic '{}'",
                final_score, topic_transition.to_topic
            );
        }

        Ok(context_preserved)
    }

    /// Get context statistics
    pub async fn get_context_stats(&self) -> ContextStats {
        ContextStats {
            total_messages: self.context_window.total_messages().await,
            active_messages: self.context_window.active_messages().await,
            pinned_messages: self.context_window.pinned_count().await,
            current_topics: self.topic_tracker.topic_count().await,
            summarization_count: self.summarization_engine.summarization_count().await,
            memory_optimizations: self.memory_optimizer.optimization_count().await,
            average_importance_score: self.importance_scorer.average_score().await,
            context_efficiency: self.calculate_context_efficiency().await,
        }
    }

    // Private helper methods

    async fn perform_summarization(&mut self) -> Result<SummarizationUpdate> {
        let messages_to_summarize = self.context_window.get_messages_for_summarization().await?;
        let summary = self
            .summarization_engine
            .summarize_messages(&messages_to_summarize)
            .await?;

        let summary_text = summary.text.clone();
        let summary_clone = summary.clone();
        self.context_window.apply_summarization(summary).await?;

        Ok(SummarizationUpdate {
            summary: summary_clone,
            messages_summarized: messages_to_summarize.len(),
            compression_ratio: self
                .calculate_compression_ratio(&messages_to_summarize, &summary_text)
                .await,
        })
    }

    async fn calculate_context_quality(&self, messages: &[Message], topics: &[Topic]) -> f32 {
        let mut quality = 0.0;

        // Message relevance
        if !messages.is_empty() {
            let relevance_sum: f32 = messages
                .iter()
                .filter_map(|m| {
                    m.metadata
                        .as_ref()
                        .and_then(|meta| meta.confidence.map(|c| c as f32))
                })
                .sum();
            quality += relevance_sum / messages.len() as f32 * 0.4;
        }

        // Topic coherence
        if !topics.is_empty() {
            let topic_confidence: f32 = topics.iter().map(|t| t.confidence).sum();
            quality += (topic_confidence / topics.len() as f32) * 0.3;
        }

        // Context completeness
        let completeness = if messages.len() >= self.config.sliding_window_size / 2 {
            1.0
        } else {
            0.5
        };
        quality += completeness * 0.3;

        quality.min(1.0)
    }

    async fn calculate_coverage_score(&self, messages: &[Message]) -> f32 {
        // Simple coverage calculation based on message diversity
        let unique_intents: std::collections::HashSet<String> = messages
            .iter()
            .filter_map(|m| {
                m.metadata
                    .as_ref()
                    .and_then(|meta| meta.custom_fields.get("intent_classification"))
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
            })
            .collect();

        if messages.is_empty() {
            0.0
        } else {
            (unique_intents.len() as f32 / messages.len() as f32).min(1.0)
        }
    }

    async fn estimate_token_count(&self, text: &str) -> usize {
        // Rough token estimation: ~4 characters per token
        text.len() / 4
    }

    async fn extract_structured_context(&self, messages: &[Message]) -> Result<StructuredContext> {
        let mut entities = Vec::new();
        let mut facts = Vec::new();
        let mut queries = Vec::new();
        let mut relationships = Vec::new();

        for message in messages {
            if let Some(metadata) = &message.metadata {
                // Extract entities from custom fields
                if let Some(extracted_entities) = metadata.custom_fields.get("entities_extracted") {
                    if let Ok(entities_list) =
                        serde_json::from_value::<Vec<String>>(extracted_entities.clone())
                    {
                        entities.extend(entities_list);
                    }
                }

                // Extract SPARQL queries from custom fields
                if let Some(sparql) = metadata.custom_fields.get("sparql_query") {
                    if let Some(query_str) = sparql.as_str() {
                        queries.push(query_str.to_string());
                    }
                }

                // Extract facts from retrieved triples in custom fields
                if let Some(triples) = metadata.custom_fields.get("retrieved_triples") {
                    if let Ok(triples_list) = serde_json::from_value::<Vec<String>>(triples.clone())
                    {
                        facts.extend(triples_list);
                    }
                }

                // Extract relationships from custom fields
                if let Some(extracted_relationships) =
                    metadata.custom_fields.get("relationships_extracted")
                {
                    if let Ok(relationships_list) =
                        serde_json::from_value::<Vec<String>>(extracted_relationships.clone())
                    {
                        relationships.extend(relationships_list);
                    }
                }

                // Extract relationships from RAG extracted relationships
                if let Some(rag_relationships) =
                    metadata.custom_fields.get("extracted_relationships")
                {
                    if let Ok(rag_relationships_list) =
                        serde_json::from_value::<Vec<String>>(rag_relationships.clone())
                    {
                        relationships.extend(rag_relationships_list);
                    }
                }

                // Extract relationships from conversation analysis
                if let Some(conversation_relationships) =
                    metadata.custom_fields.get("conversation_relationships")
                {
                    if let Ok(conversation_relationships_list) =
                        serde_json::from_value::<Vec<String>>(conversation_relationships.clone())
                    {
                        relationships.extend(conversation_relationships_list);
                    }
                }
            }
        }

        // Deduplicate relationships
        relationships.sort();
        relationships.dedup();

        // Also extract implicit relationships from facts and entities
        let implicit_relationships = self.extract_implicit_relationships(&entities, &facts).await;
        relationships.extend(implicit_relationships);

        // Final deduplication
        relationships.sort();
        relationships.dedup();

        debug!(
            "Extracted structured context: {} entities, {} facts, {} queries, {} relationships",
            entities.len(),
            facts.len(),
            queries.len(),
            relationships.len()
        );

        Ok(StructuredContext {
            entities,
            facts,
            queries,
            relationships,
        })
    }

    /// Extract implicit relationships from entities and facts
    async fn extract_implicit_relationships(
        &self,
        entities: &[String],
        facts: &[String],
    ) -> Vec<String> {
        let mut implicit_relationships = Vec::new();

        // Extract relationships from RDF facts/triples
        for fact in facts {
            if let Some(relationship) = self.parse_relationship_from_triple(fact) {
                implicit_relationships.push(relationship);
            }
        }

        // Extract relationships from entity co-occurrence patterns
        if entities.len() >= 2 {
            for i in 0..entities.len() {
                for j in (i + 1)..entities.len() {
                    let entity1 = &entities[i];
                    let entity2 = &entities[j];

                    // Create a general relationship notation
                    let relationship = format!("{} <-> {}", entity1, entity2);
                    implicit_relationships.push(relationship);
                }
            }
        }

        // Limit the number of implicit relationships to avoid explosion
        if implicit_relationships.len() > 50 {
            implicit_relationships.truncate(50);
        }

        implicit_relationships
    }

    /// Parse relationship from RDF triple format
    fn parse_relationship_from_triple(&self, triple: &str) -> Option<String> {
        // Simple regex-based parsing of RDF triples
        let patterns = [
            // Standard RDF triple: <subject> <predicate> <object>
            r"<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>",
            // With prefixes: prefix:subject prefix:predicate prefix:object
            r"(\w+:\w+)\s+(\w+:\w+)\s+(\w+:\w+)",
            // Mixed format
            r"([^\s]+)\s+([^\s]+)\s+([^\s]+)",
        ];

        for pattern in &patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                if let Some(captures) = regex.captures(triple) {
                    if captures.len() >= 4 {
                        let subject = captures.get(1)?.as_str();
                        let predicate = captures.get(2)?.as_str();
                        let object = captures.get(3)?.as_str();

                        // Clean up the predicate to make it more readable
                        let clean_predicate = predicate
                            .replace("http://", "")
                            .replace("https://", "")
                            .split('/')
                            .last()
                            .unwrap_or(predicate)
                            .replace('#', ":")
                            .to_string();

                        return Some(format!("{} --[{}]--> {}", subject, clean_predicate, object));
                    }
                }
            }
        }

        None
    }

    async fn calculate_compression_ratio(
        &self,
        original_messages: &[Message],
        summary: &str,
    ) -> f32 {
        let original_length: usize = original_messages.iter().map(|m| m.content.len()).sum();
        if original_length == 0 {
            0.0
        } else {
            summary.len() as f32 / original_length as f32
        }
    }

    async fn calculate_context_efficiency(&self) -> f32 {
        // Calculate how efficiently the context is being used
        let active_ratio = self.context_window.active_messages().await as f32
            / self.context_window.total_messages().await as f32;
        let importance_efficiency = self.importance_scorer.average_score().await;

        (active_ratio + importance_efficiency) / 2.0
    }
}

/// Context window with sliding window management
struct ContextWindow {
    config: ContextConfig,
    messages: VecDeque<ContextMessage>,
    pinned_messages: HashMap<String, PinnedMessage>,
    summary: Option<ContextSummary>,
    total_token_count: usize,
}

#[derive(Debug, Clone)]
struct ContextMessage {
    message: Message,
    importance_score: f32,
    added_at: SystemTime,
    last_accessed: SystemTime,
    access_count: usize,
}

#[derive(Debug, Clone)]
struct PinnedMessage {
    message_id: String,
    reason: PinReason,
    pinned_at: SystemTime,
    importance_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PinReason {
    HighImportance,
    UserRequest,
    KeyInformation,
    ContextAnchor,
    Reference,
}

impl ContextWindow {
    fn new(config: &ContextConfig) -> Self {
        Self {
            config: config.clone(),
            messages: VecDeque::new(),
            pinned_messages: HashMap::new(),
            summary: None,
            total_token_count: 0,
        }
    }

    async fn add_message(
        &mut self,
        message: Message,
        importance_score: f32,
    ) -> Result<WindowUpdate> {
        let context_message = ContextMessage {
            message,
            importance_score,
            added_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 1,
        };

        // Estimate token count for this message
        let message_tokens = context_message.message.content.len() / 4; // Rough estimate
        self.total_token_count += message_tokens;

        self.messages.push_back(context_message);

        // Check if we need to trim the window
        let mut evicted_messages = Vec::new();
        while self.should_trim_window() {
            if let Some(evicted) = self.evict_least_important().await? {
                evicted_messages.push(evicted);
            } else {
                break;
            }
        }

        Ok(WindowUpdate {
            message_added: true,
            evicted_messages,
            current_size: self.messages.len(),
            token_count: self.total_token_count,
        })
    }

    async fn get_effective_messages(&self) -> Result<Vec<Message>> {
        let mut effective_messages = Vec::new();

        // Add pinned messages first
        for pinned in self.pinned_messages.values() {
            if let Some(context_msg) = self
                .messages
                .iter()
                .find(|m| m.message.id == pinned.message_id)
            {
                effective_messages.push(context_msg.message.clone());
            }
        }

        // Add recent messages up to window size
        let recent_count = self
            .config
            .sliding_window_size
            .saturating_sub(effective_messages.len());
        for context_msg in self.messages.iter().rev().take(recent_count) {
            if !effective_messages
                .iter()
                .any(|m| m.id == context_msg.message.id)
            {
                effective_messages.push(context_msg.message.clone());
            }
        }

        // Sort by timestamp to maintain conversation order
        effective_messages.sort_by_key(|m| m.timestamp);

        Ok(effective_messages)
    }

    async fn pin_message(&mut self, message_id: &str, reason: PinReason) -> Result<()> {
        if let Some(context_msg) = self.messages.iter().find(|m| m.message.id == message_id) {
            let pinned = PinnedMessage {
                message_id: message_id.to_string(),
                reason,
                pinned_at: SystemTime::now(),
                importance_score: context_msg.importance_score,
            };
            self.pinned_messages.insert(message_id.to_string(), pinned);
            debug!("Pinned message: {}", message_id);
        } else {
            return Err(anyhow!(
                "Message not found in context window: {}",
                message_id
            ));
        }
        Ok(())
    }

    async fn unpin_message(&mut self, message_id: &str) -> Result<()> {
        if self.pinned_messages.remove(message_id).is_some() {
            debug!("Unpinned message: {}", message_id);
        } else {
            warn!("Attempted to unpin non-pinned message: {}", message_id);
        }
        Ok(())
    }

    fn should_trim_window(&self) -> bool {
        self.messages.len() > self.config.sliding_window_size
            || self.total_token_count > self.config.max_context_length
    }

    async fn evict_least_important(&mut self) -> Result<Option<Message>> {
        // Find least important non-pinned message
        let mut least_important_idx = None;
        let mut min_score = f32::MAX;

        for (idx, context_msg) in self.messages.iter().enumerate() {
            if !self.pinned_messages.contains_key(&context_msg.message.id)
                && context_msg.importance_score < min_score
            {
                min_score = context_msg.importance_score;
                least_important_idx = Some(idx);
            }
        }

        if let Some(idx) = least_important_idx {
            if let Some(evicted) = self.messages.remove(idx) {
                let evicted_tokens = evicted.message.content.len() / 4;
                self.total_token_count = self.total_token_count.saturating_sub(evicted_tokens);
                return Ok(Some(evicted.message));
            }
        }

        Ok(None)
    }

    async fn should_summarize(&self) -> bool {
        self.config.enable_summarization
            && self.messages.len() >= self.config.summarization_threshold
    }

    async fn get_messages_for_summarization(&self) -> Result<Vec<Message>> {
        // Get older messages that aren't pinned
        let cutoff_idx = self
            .messages
            .len()
            .saturating_sub(self.config.sliding_window_size);

        Ok(self
            .messages
            .iter()
            .take(cutoff_idx)
            .filter(|m| !self.pinned_messages.contains_key(&m.message.id))
            .map(|m| m.message.clone())
            .collect::<Vec<_>>())
    }

    async fn apply_summarization(&mut self, summary: ContextSummary) -> Result<()> {
        // Remove summarized messages
        let cutoff_idx = self
            .messages
            .len()
            .saturating_sub(self.config.sliding_window_size);
        for _ in 0..cutoff_idx {
            if let Some(removed) = self.messages.pop_front() {
                if !self.pinned_messages.contains_key(&removed.message.id) {
                    let removed_tokens = removed.message.content.len() / 4;
                    self.total_token_count = self.total_token_count.saturating_sub(removed_tokens);
                }
            }
        }

        self.summary = Some(summary);
        Ok(())
    }

    async fn get_summary(&self) -> Option<String> {
        self.summary.as_ref().map(|s| s.text.clone())
    }

    async fn total_messages(&self) -> usize {
        self.messages.len()
    }

    async fn active_messages(&self) -> usize {
        std::cmp::min(self.messages.len(), self.config.sliding_window_size)
    }

    async fn pinned_count(&self) -> usize {
        self.pinned_messages.len()
    }

    async fn get_state_snapshot(&self) -> ContextState {
        ContextState {
            message_count: self.messages.len(),
            pinned_count: self.pinned_messages.len(),
            token_count: self.total_token_count,
            has_summary: self.summary.is_some(),
            current_topic: None,
        }
    }

    async fn get_state_snapshot_with_topic(&self, current_topic: Option<String>) -> ContextState {
        ContextState {
            message_count: self.messages.len(),
            pinned_count: self.pinned_messages.len(),
            token_count: self.total_token_count,
            has_summary: self.summary.is_some(),
            current_topic,
        }
    }

    async fn adjust_for_topic(&mut self, transition: &TopicTransition) -> Result<WindowAdjustment> {
        let mut messages_reordered = false;
        let mut importance_rescored = false;
        let mut window_size_adjusted = false;

        // 1. Adjust window size based on topic complexity and confidence
        let optimal_window_size = self
            .calculate_optimal_window_size_for_topic(transition)
            .await;
        if optimal_window_size != self.config.sliding_window_size {
            self.config.sliding_window_size = optimal_window_size;
            window_size_adjusted = true;
            debug!(
                "Adjusted window size to {} for topic '{}'",
                optimal_window_size, transition.to_topic
            );
        }

        // 2. Rescore message importance based on topic relevance
        if transition.confidence > 0.6 {
            importance_rescored = self.rescore_messages_for_topic(transition).await?;
        }

        // 3. Reorder messages based on new importance scores
        if importance_rescored {
            messages_reordered = self.reorder_messages_by_importance().await;
        }

        // 4. Apply topic-specific filtering if needed
        if transition.confidence > 0.8 {
            self.apply_topic_specific_filtering(transition).await?;
        }

        // 5. Adjust message priorities for pinned messages
        self.adjust_pinned_message_priorities_for_topic(transition)
            .await?;

        info!(
            "Topic adjustments completed for '{}': reordered={}, rescored={}, window_adjusted={}",
            transition.to_topic, messages_reordered, importance_rescored, window_size_adjusted
        );

        Ok(WindowAdjustment {
            messages_reordered,
            importance_rescored,
            window_size_adjusted,
        })
    }

    /// Calculate optimal window size for the given topic
    async fn calculate_optimal_window_size_for_topic(&self, transition: &TopicTransition) -> usize {
        let base_size = self.config.sliding_window_size;

        // Adjust based on topic complexity (estimated from topic name length and confidence)
        let topic_complexity_factor = if transition.to_topic.len() > 20 {
            1.2 // Complex topics need more context
        } else if transition.to_topic.len() < 10 {
            0.8 // Simple topics need less context
        } else {
            1.0
        };

        // Adjust based on transition confidence
        let confidence_factor = if transition.confidence > 0.9 {
            1.1 // High confidence topics can use more context
        } else if transition.confidence < 0.5 {
            0.9 // Low confidence topics should use less context
        } else {
            1.0
        };

        let adjusted_size =
            (base_size as f32 * topic_complexity_factor * confidence_factor) as usize;

        // Clamp to reasonable bounds
        adjusted_size.max(10).min(100)
    }

    /// Rescore messages based on their relevance to the new topic
    async fn rescore_messages_for_topic(&mut self, transition: &TopicTransition) -> Result<bool> {
        let mut rescored = false;
        let topic_keywords = self.extract_topic_keywords(&transition.to_topic);

        // Create a vector of adjustments to avoid borrow checker issues
        let mut adjustments = Vec::new();

        for (index, context_message) in self.messages.iter().enumerate() {
            let topic_relevance = Self::calculate_message_topic_relevance_static(
                &context_message.message,
                &topic_keywords,
            );

            let original_score = context_message.importance_score;
            let topic_adjustment = match topic_relevance {
                relevance if relevance > 0.8 => 1.3, // High relevance boost
                relevance if relevance > 0.5 => 1.1, // Moderate relevance boost
                relevance if relevance > 0.2 => 1.0, // No change
                _ => 0.8,                            // Low relevance penalty
            };

            let new_score = (original_score * topic_adjustment).min(1.0);
            adjustments.push((index, new_score, original_score));
        }

        // Apply adjustments
        for (index, new_score, original_score) in adjustments {
            if let Some(context_message) = self.messages.get_mut(index) {
                context_message.importance_score = new_score;
                if (new_score - original_score).abs() > 0.05 {
                    rescored = true;
                }
            }
        }

        if rescored {
            debug!(
                "Rescored {} messages for topic relevance",
                self.messages.len()
            );
        }

        Ok(rescored)
    }

    /// Extract keywords from topic name for relevance calculation
    fn extract_topic_keywords(&self, topic: &str) -> Vec<String> {
        topic
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| word.len() > 2) // Filter out short words
            .map(|word| word.to_string())
            .collect()
    }

    /// Calculate how relevant a message is to the current topic
    async fn calculate_message_topic_relevance(
        &self,
        message: &Message,
        topic_keywords: &[String],
    ) -> f32 {
        Self::calculate_message_topic_relevance_static(message, topic_keywords)
    }

    /// Static version to avoid borrow checker issues
    fn calculate_message_topic_relevance_static(
        message: &Message,
        topic_keywords: &[String],
    ) -> f32 {
        let message_text = message.content.to_lowercase();
        let mut relevance_score = 0.0;
        let mut keyword_matches = 0;

        for keyword in topic_keywords {
            if message_text.contains(keyword) {
                keyword_matches += 1;
                relevance_score += 0.2; // Base score per keyword match

                // Bonus for exact word matches (not just substring matches)
                if message_text.split_whitespace().any(|word| word == keyword) {
                    relevance_score += 0.1;
                }
            }
        }

        // Apply diminishing returns for multiple keyword matches
        if keyword_matches > 0 {
            relevance_score = relevance_score * (1.0 - (keyword_matches as f32 * 0.05).min(0.3));
        }

        // Check message metadata for additional topic relevance indicators
        if let Some(_metadata) = &message.metadata {
            // Basic boost for messages with metadata (indicating they were processed)
            relevance_score += 0.05;
        }

        relevance_score.min(1.0)
    }

    /// Reorder messages by their importance scores
    async fn reorder_messages_by_importance(&mut self) -> bool {
        let original_order: Vec<_> = self.messages.iter().map(|m| m.message.id.clone()).collect();

        // Sort by importance score (descending) while maintaining relative chronological order for equal scores
        self.messages.make_contiguous().sort_by(|a, b| {
            b.importance_score
                .partial_cmp(&a.importance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.added_at.cmp(&b.added_at))
        });

        let new_order: Vec<_> = self.messages.iter().map(|m| m.message.id.clone()).collect();
        let reordered = original_order != new_order;

        if reordered {
            debug!("Reordered {} messages by importance", self.messages.len());
        }

        reordered
    }

    /// Apply topic-specific filtering to remove less relevant messages
    async fn apply_topic_specific_filtering(&mut self, transition: &TopicTransition) -> Result<()> {
        let original_count = self.messages.len();

        // Remove messages with very low importance scores (below threshold)
        let min_importance_threshold = if transition.confidence > 0.9 {
            0.3
        } else {
            0.2
        };

        self.messages
            .retain(|context_message| context_message.importance_score >= min_importance_threshold);

        let filtered_count = original_count - self.messages.len();
        if filtered_count > 0 {
            debug!(
                "Filtered out {} low-importance messages for topic '{}'",
                filtered_count, transition.to_topic
            );
        }

        Ok(())
    }

    /// Adjust priorities for pinned messages based on topic relevance
    async fn adjust_pinned_message_priorities_for_topic(
        &mut self,
        transition: &TopicTransition,
    ) -> Result<()> {
        let topic_keywords = self.extract_topic_keywords(&transition.to_topic);

        // Collect adjustments to avoid borrow checker issues
        let mut adjustments = Vec::new();

        for (message_id, pinned_message) in &self.pinned_messages {
            if let Some(context_message) =
                self.messages.iter().find(|m| m.message.id == *message_id)
            {
                let topic_relevance = Self::calculate_message_topic_relevance_static(
                    &context_message.message,
                    &topic_keywords,
                );

                let original_score = pinned_message.importance_score;
                let new_score = (original_score + topic_relevance * 0.3).min(1.0);

                adjustments.push((message_id.clone(), original_score, new_score));
            }
        }

        // Apply adjustments
        for (message_id, original_score, new_score) in adjustments {
            if let Some(pinned_message) = self.pinned_messages.get_mut(&message_id) {
                pinned_message.importance_score = new_score;

                debug!(
                    "Adjusted pinned message '{}' importance from {:.2} to {:.2} for topic relevance",
                    message_id, original_score, new_score
                );
            }
        }

        Ok(())
    }
}

/// Supporting structures and types

#[derive(Debug, Clone)]
pub struct AssembledContext {
    pub context_text: String,
    pub effective_messages: Vec<Message>,
    pub current_topics: Vec<Topic>,
    pub context_summary: Option<String>,
    pub quality_score: f32,
    pub coverage_score: f32,
    pub token_count: usize,
    pub structured_context: StructuredContext,
}

#[derive(Debug, Clone)]
pub struct StructuredContext {
    pub entities: Vec<String>,
    pub facts: Vec<String>,
    pub queries: Vec<String>,
    pub relationships: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Topic {
    pub name: String,
    pub confidence: f32,
    pub first_mentioned: SystemTime,
    pub last_mentioned: SystemTime,
    pub mention_count: usize,
}

#[derive(Debug, Clone)]
pub struct ContextUpdate {
    pub message_processed: String,
    pub importance_score: f32,
    pub window_update: WindowUpdate,
    pub topic_update: Option<TopicUpdate>,
    pub summarization_update: Option<SummarizationUpdate>,
    pub optimization_update: Option<OptimizationUpdate>,
    pub processing_time: Duration,
}

#[derive(Debug, Clone)]
pub struct WindowUpdate {
    pub message_added: bool,
    pub evicted_messages: Vec<Message>,
    pub current_size: usize,
    pub token_count: usize,
}

#[derive(Debug, Clone)]
pub struct TopicUpdate {
    pub new_topics: Vec<Topic>,
    pub topic_changes: Vec<TopicChange>,
    pub drift_detected: bool,
}

#[derive(Debug, Clone)]
pub struct TopicChange {
    pub topic_name: String,
    pub change_type: TopicChangeType,
    pub confidence_delta: f32,
}

#[derive(Debug, Clone)]
pub enum TopicChangeType {
    Introduced,
    Strengthened,
    Weakened,
    Abandoned,
}

#[derive(Debug, Clone)]
pub struct SummarizationUpdate {
    pub summary: ContextSummary,
    pub messages_summarized: usize,
    pub compression_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct ContextSummary {
    pub text: String,
    pub key_points: Vec<String>,
    pub entities_mentioned: Vec<String>,
    pub topics_covered: Vec<String>,
    pub created_at: SystemTime,
}

#[derive(Debug, Clone)]
pub struct OptimizationUpdate {
    pub memory_saved: usize,
    pub operations_performed: Vec<OptimizationOperation>,
    pub efficiency_improvement: f32,
}

#[derive(Debug, Clone)]
pub enum OptimizationOperation {
    MessageDeduplication,
    ImportanceRescoring,
    TokenCompression,
    RedundancyRemoval,
}

#[derive(Debug, Clone)]
pub struct ContextSwitch {
    pub previous_state: ContextState,
    pub new_topic: String,
    pub topic_transition: TopicTransition,
    pub window_adjustment: WindowAdjustment,
    pub context_preserved: bool,
}

#[derive(Debug, Clone)]
pub struct ContextState {
    pub message_count: usize,
    pub pinned_count: usize,
    pub token_count: usize,
    pub has_summary: bool,
    pub current_topic: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TopicTransition {
    pub from_topic: Option<String>,
    pub to_topic: String,
    pub transition_reason: String,
    pub confidence: f32,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone)]
pub struct WindowAdjustment {
    pub messages_reordered: bool,
    pub importance_rescored: bool,
    pub window_size_adjusted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextStats {
    pub total_messages: usize,
    pub active_messages: usize,
    pub pinned_messages: usize,
    pub current_topics: usize,
    pub summarization_count: usize,
    pub memory_optimizations: usize,
    pub average_importance_score: f32,
    pub context_efficiency: f32,
}
