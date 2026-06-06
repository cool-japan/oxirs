//! Memory operation systems: MemoryNetworks, EpisodicMemory, RelationalMemoryCore,
//! SparseAccessMemory, and the top-level MemoryAugmentedNetwork orchestrator.

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{s, Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::info;
use uuid::Uuid;

use crate::memory_nets_controller::{
    DNCConfig, DifferentiableNeuralComputer, NTMConfig, NeuralTuringMachine,
};

/// Memory Networks configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNetworksConfig {
    pub memory_capacity: usize,
    pub embedding_dim: usize,
    pub num_hops: usize,
    pub learning_rate: f32,
}

impl Default for MemoryNetworksConfig {
    fn default() -> Self {
        Self {
            memory_capacity: 1000,
            embedding_dim: 128,
            num_hops: 3,
            learning_rate: 0.01,
        }
    }
}

/// Memory Networks implementation (multi-hop reasoning)
pub struct MemoryNetworks {
    pub(crate) config: MemoryNetworksConfig,
    pub(crate) memory_embeddings: Array2<f32>,
    pub(crate) memory_content: Vec<String>,
    pub(crate) input_encoder: Array2<f32>,
    pub(crate) output_encoder: Array2<f32>,
    pub(crate) query_encoder: Array2<f32>,
}

impl MemoryNetworks {
    pub fn new(config: MemoryNetworksConfig) -> Self {
        use scirs2_core::random::Random;
        let mut rng = Random::default();

        let memory_embeddings = Array2::zeros((config.memory_capacity, config.embedding_dim));
        let memory_content = Vec::new();

        let input_encoder =
            Array2::from_shape_fn((config.embedding_dim, config.embedding_dim), |_| {
                rng.random_range(-0.1..0.1)
            });
        let output_encoder =
            Array2::from_shape_fn((config.embedding_dim, config.embedding_dim), |_| {
                rng.random_range(-0.1..0.1)
            });
        let query_encoder =
            Array2::from_shape_fn((config.embedding_dim, config.embedding_dim), |_| {
                rng.random_range(-0.1..0.1)
            });

        Self {
            config,
            memory_embeddings,
            memory_content,
            input_encoder,
            output_encoder,
            query_encoder,
        }
    }

    /// Store memory (FIFO eviction when full)
    pub fn store_memory(&mut self, content: String, embedding: Array1<f32>) -> Result<()> {
        if self.memory_content.len() < self.config.memory_capacity {
            let index = self.memory_content.len();
            self.memory_content.push(content);
            if embedding.len() == self.config.embedding_dim {
                self.memory_embeddings.row_mut(index).assign(&embedding);
            } else {
                return Err(anyhow!("Embedding dimension mismatch"));
            }
        } else {
            let index = 0;
            self.memory_content[index] = content;
            self.memory_embeddings.row_mut(index).assign(&embedding);
            for i in 1..self.memory_content.len() {
                self.memory_content.swap(i - 1, i);
                let row1 = self.memory_embeddings.row(i - 1).to_owned();
                let row2 = self.memory_embeddings.row(i).to_owned();
                self.memory_embeddings.row_mut(i - 1).assign(&row2);
                self.memory_embeddings.row_mut(i).assign(&row1);
            }
        }
        Ok(())
    }

    /// Query memory using multi-hop attention
    pub fn query(&self, query_embedding: &Array1<f32>) -> Result<Array1<f32>> {
        let num_memories = self.memory_content.len();
        if num_memories == 0 {
            return Ok(Array1::zeros(self.config.embedding_dim));
        }

        let mut response = Array1::zeros(self.config.embedding_dim);
        let mut current_query = query_embedding.clone();

        for _hop in 0..self.config.num_hops {
            let attention_weights = self.compute_attention(&current_query)?;
            // Only use the filled portion of the embeddings matrix
            let active_embeddings = self
                .memory_embeddings
                .slice(scirs2_core::ndarray_ext::s![..num_memories, ..]);
            let memory_response = active_embeddings.t().dot(&attention_weights);
            current_query = self.output_encoder.dot(&memory_response);
            response = memory_response;
        }

        Ok(response)
    }

    fn compute_attention(&self, query: &Array1<f32>) -> Result<Array1<f32>> {
        let num_memories = self.memory_content.len();
        if num_memories == 0 {
            return Ok(Array1::zeros(0));
        }

        let mut attention_scores = Array1::zeros(num_memories);
        for i in 0..num_memories {
            let memory_embedding = self.memory_embeddings.row(i);
            attention_scores[i] = query.dot(&memory_embedding);
        }

        let max_score = attention_scores.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores = attention_scores.map(|&x| (x - max_score).exp());
        let sum_exp = exp_scores.sum();

        if sum_exp > 0.0 {
            Ok(exp_scores / sum_exp)
        } else {
            Ok(Array1::from_elem(num_memories, 1.0 / num_memories as f32))
        }
    }
}

/// Episodic Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicConfig {
    pub episode_capacity: usize,
    pub episode_length: usize,
    pub embedding_dim: usize,
    pub decay_factor: f32,
}

impl Default for EpisodicConfig {
    fn default() -> Self {
        Self {
            episode_capacity: 100,
            episode_length: 50,
            embedding_dim: 128,
            decay_factor: 0.95,
        }
    }
}

/// Episode metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeMetadata {
    pub episode_type: String,
    pub success: bool,
    pub length: usize,
    pub average_reward: f32,
    pub tags: Vec<String>,
}

/// Episode representation
#[derive(Debug, Clone)]
pub struct Episode {
    pub id: Uuid,
    pub states: Vec<Array1<f32>>,
    pub rewards: Vec<f32>,
    pub metadata: EpisodeMetadata,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Episodic Memory for sequential experiences
pub struct EpisodicMemory {
    pub(crate) config: EpisodicConfig,
    pub(crate) episodes: VecDeque<Episode>,
    pub(crate) current_episode: Option<Episode>,
}

impl EpisodicMemory {
    pub fn new(config: EpisodicConfig) -> Self {
        Self {
            config,
            episodes: VecDeque::new(),
            current_episode: None,
        }
    }

    pub fn start_episode(&mut self, episode_type: String) {
        let episode = Episode {
            id: Uuid::new_v4(),
            states: Vec::new(),
            rewards: Vec::new(),
            metadata: EpisodeMetadata {
                episode_type,
                success: false,
                length: 0,
                average_reward: 0.0,
                tags: Vec::new(),
            },
            timestamp: chrono::Utc::now(),
        };
        self.current_episode = Some(episode);
    }

    pub fn add_state(&mut self, state: Array1<f32>, reward: f32) -> Result<()> {
        if let Some(ref mut episode) = self.current_episode {
            episode.states.push(state);
            episode.rewards.push(reward);
            Ok(())
        } else {
            Err(anyhow!("No active episode"))
        }
    }

    pub fn end_episode(&mut self, success: bool) -> Result<()> {
        if let Some(mut episode) = self.current_episode.take() {
            episode.metadata.success = success;
            episode.metadata.length = episode.states.len();
            episode.metadata.average_reward = if episode.rewards.is_empty() {
                0.0
            } else {
                episode.rewards.iter().sum::<f32>() / episode.rewards.len() as f32
            };

            if self.episodes.len() >= self.config.episode_capacity {
                self.episodes.pop_front();
            }
            self.episodes.push_back(episode);
            Ok(())
        } else {
            Err(anyhow!("No active episode"))
        }
    }

    pub fn retrieve_similar_episodes(&self, query_state: &Array1<f32>, k: usize) -> Vec<&Episode> {
        let mut similarities: Vec<(f32, &Episode)> = self
            .episodes
            .iter()
            .map(|episode| {
                let similarity = self.compute_episode_similarity(episode, query_state);
                (similarity, episode)
            })
            .collect();

        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        similarities
            .into_iter()
            .take(k)
            .map(|(_, episode)| episode)
            .collect()
    }

    fn compute_episode_similarity(&self, episode: &Episode, query_state: &Array1<f32>) -> f32 {
        if episode.states.is_empty() {
            return 0.0;
        }
        let mut total_similarity = 0.0;
        for state in &episode.states {
            total_similarity += cosine_sim(query_state, state);
        }
        total_similarity / episode.states.len() as f32
    }
}

/// Relational Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationalConfig {
    pub memory_size: usize,
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub num_relation_types: usize,
}

impl Default for RelationalConfig {
    fn default() -> Self {
        Self {
            memory_size: 512,
            embedding_dim: 256,
            num_heads: 8,
            num_relation_types: 10,
        }
    }
}

/// Relational attention mechanism
pub struct RelationalAttention {
    pub(crate) query_weights: Array2<f32>,
    pub(crate) key_weights: Array2<f32>,
    pub(crate) value_weights: Array2<f32>,
    pub(crate) num_heads: usize,
    pub(crate) embed_dim: usize,
}

impl RelationalAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        use scirs2_core::random::Random;
        let mut rng = Random::default();

        let query_weights =
            Array2::from_shape_fn((embed_dim, embed_dim), |_| rng.random_range(-0.1..0.1));
        let key_weights =
            Array2::from_shape_fn((embed_dim, embed_dim), |_| rng.random_range(-0.1..0.1));
        let value_weights =
            Array2::from_shape_fn((embed_dim, embed_dim), |_| rng.random_range(-0.1..0.1));

        Self {
            query_weights,
            key_weights,
            value_weights,
            num_heads,
            embed_dim,
        }
    }

    pub fn forward(&self, memory: &Array2<f32>, query: &Array1<f32>) -> Array1<f32> {
        let head_dim = self.embed_dim / self.num_heads;
        let mut output = Array1::zeros(self.embed_dim);

        for head in 0..self.num_heads {
            let start_idx = head * head_dim;
            let end_idx = (head + 1) * head_dim;

            let q_head = self.query_weights.slice(s![start_idx..end_idx, ..]);
            let k_head = self.key_weights.slice(s![start_idx..end_idx, ..]);
            let v_head = self.value_weights.slice(s![start_idx..end_idx, ..]);

            let q = q_head.dot(query);
            let keys = memory.dot(&k_head.t());
            let values = memory.dot(&v_head.t());

            let mut scores = Array1::zeros(memory.nrows());
            for i in 0..memory.nrows() {
                scores[i] = q.dot(&keys.row(i));
            }

            let max_score = scores.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_scores = scores.map(|&x| (x - max_score).exp());
            let sum_exp = exp_scores.sum();
            let attention_weights = if sum_exp > 0.0 {
                exp_scores / sum_exp
            } else {
                Array1::from_elem(memory.nrows(), 1.0 / memory.nrows() as f32)
            };

            let head_output = values.t().dot(&attention_weights);
            output
                .slice_mut(s![start_idx..end_idx])
                .assign(&head_output);
        }

        output
    }
}

/// Relational Memory Core for structured knowledge
pub struct RelationalMemoryCore {
    pub(crate) config: RelationalConfig,
    pub(crate) memory: Array2<f32>,
    pub(crate) relation_matrices: Vec<Array2<f32>>,
    pub(crate) attention_mechanism: RelationalAttention,
}

impl RelationalMemoryCore {
    pub fn new(config: RelationalConfig) -> Self {
        use scirs2_core::random::Random;
        let mut rng = Random::default();

        let memory = Array2::zeros((config.memory_size, config.embedding_dim));
        let mut relation_matrices = Vec::new();

        for _ in 0..config.num_relation_types {
            let relation_matrix =
                Array2::from_shape_fn((config.embedding_dim, config.embedding_dim), |_| {
                    rng.random_range(-0.1..0.1)
                });
            relation_matrices.push(relation_matrix);
        }

        let attention_mechanism = RelationalAttention::new(config.embedding_dim, config.num_heads);
        Self {
            config,
            memory,
            relation_matrices,
            attention_mechanism,
        }
    }

    pub fn store_relation(
        &mut self,
        subject: &Array1<f32>,
        relation_type: usize,
        object: &Array1<f32>,
    ) -> Result<()> {
        if relation_type >= self.config.num_relation_types {
            return Err(anyhow!("Invalid relation type"));
        }
        let relation_matrix = &self.relation_matrices[relation_type];
        let transformed_subject = relation_matrix.dot(subject);
        let transformed_object = relation_matrix.dot(object);

        if let Some(slot) = self.find_empty_slot() {
            let combined = &transformed_subject + &transformed_object;
            self.memory.row_mut(slot).assign(&combined);
        }
        Ok(())
    }

    fn find_empty_slot(&self) -> Option<usize> {
        (0..self.memory.nrows()).find(|&i| self.memory.row(i).sum() == 0.0)
    }

    pub fn query_relations(&self, query: &Array1<f32>) -> Array1<f32> {
        self.attention_mechanism.forward(&self.memory, query)
    }
}

/// Sparse Access Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseConfig {
    pub memory_capacity: usize,
    pub embedding_dim: usize,
    pub sparsity_factor: f32,
    pub update_threshold: f32,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            memory_capacity: 10000,
            embedding_dim: 512,
            sparsity_factor: 0.1,
            update_threshold: 0.01,
        }
    }
}

/// Sparse Access Memory for large-scale memory
pub struct SparseAccessMemory {
    pub(crate) config: SparseConfig,
    pub(crate) memory: HashMap<usize, Array1<f32>>,
    pub(crate) access_counts: HashMap<usize, usize>,
    pub(crate) last_access: HashMap<usize, Instant>,
}

impl SparseAccessMemory {
    pub fn new(config: SparseConfig) -> Self {
        Self {
            config,
            memory: HashMap::new(),
            access_counts: HashMap::new(),
            last_access: HashMap::new(),
        }
    }

    pub fn store(&mut self, key: usize, value: Array1<f32>) -> Result<()> {
        if self.memory.len() >= self.config.memory_capacity {
            self.evict_least_used()?;
        }
        self.memory.insert(key, value);
        self.access_counts.insert(key, 1);
        self.last_access.insert(key, Instant::now());
        Ok(())
    }

    pub fn retrieve(&mut self, key: usize) -> Option<&Array1<f32>> {
        if let Some(value) = self.memory.get(&key) {
            *self.access_counts.entry(key).or_insert(0) += 1;
            self.last_access.insert(key, Instant::now());
            Some(value)
        } else {
            None
        }
    }

    pub fn find_similar(&self, query: &Array1<f32>, k: usize) -> Vec<(usize, f32)> {
        let mut similarities: Vec<(usize, f32)> = self
            .memory
            .iter()
            .map(|(&key, value)| (key, cosine_sim(query, value)))
            .collect();
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.into_iter().take(k).collect()
    }

    fn evict_least_used(&mut self) -> Result<()> {
        let mut candidates: Vec<(usize, usize, Instant)> = self
            .access_counts
            .iter()
            .map(|(&key, &count)| {
                let last_access = self
                    .last_access
                    .get(&key)
                    .copied()
                    .unwrap_or(Instant::now());
                (key, count, last_access)
            })
            .collect();
        candidates.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2)));

        if let Some((key_to_evict, _, _)) = candidates.first() {
            let key = *key_to_evict;
            self.memory.remove(&key);
            self.access_counts.remove(&key);
            self.last_access.remove(&key);
        }
        Ok(())
    }

    pub fn cleanup(&mut self, max_age: Duration) -> Result<usize> {
        let now = Instant::now();
        let mut keys_to_remove = Vec::new();

        for (&key, &last_access) in &self.last_access {
            if now.duration_since(last_access) > max_age {
                keys_to_remove.push(key);
            }
        }

        let removed_count = keys_to_remove.len();
        for key in keys_to_remove {
            self.memory.remove(&key);
            self.access_counts.remove(&key);
            self.last_access.remove(&key);
        }
        Ok(removed_count)
    }
}

/// Coordination strategy for multiple memory systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationStrategy {
    RoundRobin,
    PerformanceBased,
    ContentBased,
    Adaptive,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageStats {
    pub dnc_utilization: f32,
    pub ntm_utilization: f32,
    pub memory_networks_utilization: f32,
    pub episodic_utilization: f32,
    pub relational_utilization: f32,
    pub sparse_utilization: f32,
    pub total_memory_mb: f32,
}

/// Performance tracker for memory systems
#[derive(Default)]
pub struct MemoryPerformanceTracker {
    pub(crate) access_latencies: HashMap<String, VecDeque<f32>>,
    pub(crate) hit_rates: HashMap<String, f32>,
    pub(crate) throughput_metrics: HashMap<String, f32>,
}

impl MemoryPerformanceTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_access(&mut self, memory_type: &str, latency_ms: f32) {
        let latencies = self
            .access_latencies
            .entry(memory_type.to_string())
            .or_default();
        latencies.push_back(latency_ms);
        while latencies.len() > 100 {
            latencies.pop_front();
        }
    }

    pub fn get_average_latency(&self, memory_type: &str) -> f32 {
        if let Some(latencies) = self.access_latencies.get(memory_type) {
            if !latencies.is_empty() {
                return latencies.iter().sum::<f32>() / latencies.len() as f32;
            }
        }
        0.0
    }
}

/// Memory coordination system
pub struct MemoryCoordinator {
    pub(crate) strategy: CoordinationStrategy,
    pub(crate) usage_stats: MemoryUsageStats,
    pub(crate) performance_tracker: MemoryPerformanceTracker,
}

/// Memory performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPerformanceMetrics {
    pub total_operations: u64,
    pub average_latency_ms: f32,
    pub hit_rate: f32,
    pub utilization: f32,
    pub ops_per_second: f32,
    pub error_rate: f32,
}

impl Default for MemoryPerformanceMetrics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            average_latency_ms: 0.0,
            hit_rate: 0.0,
            utilization: 0.0,
            ops_per_second: 0.0,
            error_rate: 0.0,
        }
    }
}

/// Global memory system settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMemorySettings {
    pub enable_compression: bool,
    pub memory_capacity_mb: f32,
    pub cleanup_threshold: f32,
    pub enable_persistence: bool,
    pub update_frequency_ms: u64,
    pub enable_coordination: bool,
}

impl Default for GlobalMemorySettings {
    fn default() -> Self {
        Self {
            enable_compression: true,
            memory_capacity_mb: 1024.0,
            cleanup_threshold: 0.85,
            enable_persistence: true,
            update_frequency_ms: 100,
            enable_coordination: true,
        }
    }
}

/// Configuration for memory-augmented networks
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryConfig {
    pub dnc_config: DNCConfig,
    pub ntm_config: NTMConfig,
    pub memory_networks_config: MemoryNetworksConfig,
    pub episodic_config: EpisodicConfig,
    pub relational_config: RelationalConfig,
    pub sparse_config: SparseConfig,
    pub global_settings: GlobalMemorySettings,
}

/// Memory-Augmented Network Engine
pub struct MemoryAugmentedNetwork {
    pub(crate) config: MemoryConfig,
    pub(crate) dnc: DifferentiableNeuralComputer,
    pub(crate) ntm: NeuralTuringMachine,
    pub(crate) memory_networks: MemoryNetworks,
    pub(crate) episodic_memory: EpisodicMemory,
    pub(crate) relational_memory: RelationalMemoryCore,
    pub(crate) sparse_memory: SparseAccessMemory,
    pub(crate) memory_coordinator: MemoryCoordinator,
    pub(crate) performance_metrics: MemoryPerformanceMetrics,
}

impl MemoryAugmentedNetwork {
    pub fn new(config: MemoryConfig) -> Result<Self> {
        let dnc = DifferentiableNeuralComputer::new(config.dnc_config.clone());
        let ntm = NeuralTuringMachine::new(config.ntm_config.clone());
        let memory_networks = MemoryNetworks::new(config.memory_networks_config.clone());
        let episodic_memory = EpisodicMemory::new(config.episodic_config.clone());
        let relational_memory = RelationalMemoryCore::new(config.relational_config.clone());
        let sparse_memory = SparseAccessMemory::new(config.sparse_config.clone());

        let memory_coordinator = MemoryCoordinator {
            strategy: CoordinationStrategy::Adaptive,
            usage_stats: MemoryUsageStats {
                dnc_utilization: 0.0,
                ntm_utilization: 0.0,
                memory_networks_utilization: 0.0,
                episodic_utilization: 0.0,
                relational_utilization: 0.0,
                sparse_utilization: 0.0,
                total_memory_mb: 0.0,
            },
            performance_tracker: MemoryPerformanceTracker::new(),
        };

        Ok(Self {
            config,
            dnc,
            ntm,
            memory_networks,
            episodic_memory,
            relational_memory,
            sparse_memory,
            memory_coordinator,
            performance_metrics: MemoryPerformanceMetrics::default(),
        })
    }

    pub async fn process(
        &mut self,
        input: &Array1<f32>,
        memory_type: Option<&str>,
    ) -> Result<Array1<f32>> {
        let start_time = Instant::now();

        let result = match memory_type {
            Some("dnc") => self.dnc.forward(input),
            Some("ntm") => self.ntm.forward(input),
            Some("memory_networks") => Ok(self.memory_networks.query(input)?),
            Some("relational") => Ok(self.relational_memory.query_relations(input)),
            Some("sparse") => {
                let similar = self.sparse_memory.find_similar(input, 1);
                if let Some((key, _)) = similar.first() {
                    Ok(self.sparse_memory.retrieve(*key).unwrap_or(input).clone())
                } else {
                    Ok(input.clone())
                }
            }
            _ => self.adaptive_routing(input).await,
        };

        let latency = start_time.elapsed().as_millis() as f32;
        if let Some(mem_type) = memory_type {
            self.memory_coordinator
                .performance_tracker
                .record_access(mem_type, latency);
        }

        self.performance_metrics.total_operations += 1;
        self.update_performance_metrics(latency);

        result
    }

    async fn adaptive_routing(&mut self, input: &Array1<f32>) -> Result<Array1<f32>> {
        let input_norm = input.mapv(|x| x * x).sum().sqrt();
        let input_sparsity =
            input.iter().filter(|&&x| x.abs() < 0.01).count() as f32 / input.len() as f32;

        match (input_norm, input_sparsity) {
            (norm, sparsity) if norm > 10.0 && sparsity < 0.3 => self.dnc.forward(input),
            (norm, sparsity) if norm < 5.0 && sparsity > 0.7 => {
                let similar = self.sparse_memory.find_similar(input, 1);
                if let Some((key, _)) = similar.first() {
                    Ok(self.sparse_memory.retrieve(*key).unwrap_or(input).clone())
                } else {
                    Ok(input.clone())
                }
            }
            _ => Ok(self.memory_networks.query(input)?),
        }
    }

    pub async fn store(
        &mut self,
        content: String,
        embedding: Array1<f32>,
        memory_type: Option<&str>,
    ) -> Result<()> {
        match memory_type {
            Some("memory_networks") => {
                self.memory_networks.store_memory(content, embedding)?;
            }
            Some("sparse") => {
                let key = self.hash_content(&content);
                self.sparse_memory.store(key, embedding)?;
            }
            Some("relational") => {
                let zero_vector = Array1::zeros(embedding.len());
                self.relational_memory
                    .store_relation(&embedding, 0, &zero_vector)?;
            }
            _ => {
                self.memory_networks.store_memory(content, embedding)?;
            }
        }
        Ok(())
    }

    fn hash_content(&self, content: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish() as usize
    }

    pub fn start_episode(&mut self, episode_type: String) {
        self.episodic_memory.start_episode(episode_type);
    }

    pub fn add_episode_state(&mut self, state: Array1<f32>, reward: f32) -> Result<()> {
        self.episodic_memory.add_state(state, reward)
    }

    pub fn end_episode(&mut self, success: bool) -> Result<()> {
        self.episodic_memory.end_episode(success)
    }

    pub fn get_memory_stats(&self) -> MemoryUsageStats {
        self.memory_coordinator.usage_stats.clone()
    }

    pub fn get_performance_metrics(&self) -> &MemoryPerformanceMetrics {
        &self.performance_metrics
    }

    fn update_performance_metrics(&mut self, latency: f32) {
        let alpha = 0.1;
        self.performance_metrics.average_latency_ms =
            alpha * latency + (1.0 - alpha) * self.performance_metrics.average_latency_ms;
    }

    pub async fn cleanup(&mut self) -> Result<()> {
        if self.dnc.get_memory_utilization() > 0.9 {
            self.dnc.reset();
        }
        let cleanup_duration = Duration::from_secs(3600);
        let removed = self.sparse_memory.cleanup(cleanup_duration)?;
        if removed > 0 {
            info!("Cleaned up {} entries from sparse memory", removed);
        }
        Ok(())
    }
}

fn cosine_sim(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot_product = a.dot(b);
    let norm_a = a.mapv(|x| x * x).sum().sqrt();
    let norm_b = b.mapv(|x| x * x).sum().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}
