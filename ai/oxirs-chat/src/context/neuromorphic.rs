//! Neuromorphic Context Processing Engine
//!
//! Implements brain-inspired processing patterns for context understanding.

use super::config::ContextConfig;
use crate::Message;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

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
            dendrites.push(Dendrite::new(format!("dendrite_{i}")));
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
        let _axonal_output = if activation {
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
            let signal = dendrite.process_input(input.features.get(i).unwrap_or(&0.0));
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

impl Default for Soma {
    fn default() -> Self {
        Self::new()
    }
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
        self.membrane_potential *= 1.0 - self.leak_conductance;

        self.membrane_potential
    }
}

#[derive(Debug, Clone)]
pub struct Axon {
    pub conduction_velocity: f64,
    pub myelination: f64,
    pub branching_factor: usize,
}

impl Default for Axon {
    fn default() -> Self {
        Self::new()
    }
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
        let transmission_delay = Duration::from_millis(((1.0 - self.myelination) * 10.0) as u64);

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
        // Create specialized processors for different context aspects
        let processors = vec![
            NeuromorphicProcessor::new(10), // Semantic processor
            NeuromorphicProcessor::new(8),  // Temporal processor
            NeuromorphicProcessor::new(6),  // Emotional processor
            NeuromorphicProcessor::new(12), // Contextual processor
        ];

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

impl Default for ContextFeatures {
    fn default() -> Self {
        Self::new()
    }
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

impl Default for AttentionMechanism {
    fn default() -> Self {
        Self::new()
    }
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
                    _ => format!("Processor {i}"),
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

    fn calculate_performance_improvement(&self, _output: &NeuromorphicOutput) -> f64 {
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

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
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
