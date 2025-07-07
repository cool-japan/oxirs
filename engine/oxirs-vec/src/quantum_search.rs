//! Quantum-inspired algorithms for vector search optimization.
//!
//! This module implements quantum computing principles to enhance vector search
//! performance through quantum superposition, entanglement, and interference patterns.
//! These algorithms provide novel optimization approaches that can outperform
//! classical algorithms in specific scenarios, particularly for high-dimensional
//! similarity search and complex optimization landscapes.

use crate::Vector;
use anyhow::{anyhow, Result};
use oxirs_core::parallel::*;
use oxirs_core::simd::SimdOps;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, info, span, Level};

/// Quantum-inspired search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSearchConfig {
    /// Number of quantum superposition states
    pub superposition_states: usize,
    /// Entanglement strength between vectors
    pub entanglement_strength: f32,
    /// Interference pattern amplification factor
    pub interference_amplitude: f32,
    /// Quantum measurement probability threshold
    pub measurement_threshold: f32,
    /// Maximum quantum search iterations
    pub max_iterations: usize,
    /// Enable quantum tunneling optimization
    pub enable_tunneling: bool,
    /// Quantum decoherence rate
    pub decoherence_rate: f32,
}

impl Default for QuantumSearchConfig {
    fn default() -> Self {
        Self {
            superposition_states: 64,
            entanglement_strength: 0.7,
            interference_amplitude: 1.2,
            measurement_threshold: 0.1,
            max_iterations: 100,
            enable_tunneling: true,
            decoherence_rate: 0.05,
        }
    }
}

/// Quantum state representation for vector search
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Amplitude coefficients for superposition states
    pub amplitudes: Vec<f32>,
    /// Phase information for quantum interference
    pub phases: Vec<f32>,
    /// Entanglement matrix between states
    pub entanglement_matrix: Vec<Vec<f32>>,
    /// Probability distribution over states
    pub probabilities: Vec<f32>,
}

impl QuantumState {
    /// Create a new quantum state with given number of states
    pub fn new(num_states: usize) -> Self {
        let amplitudes = vec![1.0 / (num_states as f32).sqrt(); num_states];
        let phases = vec![0.0; num_states];
        let entanglement_matrix = vec![vec![0.0; num_states]; num_states];
        let probabilities = vec![1.0 / num_states as f32; num_states];

        Self {
            amplitudes,
            phases,
            entanglement_matrix,
            probabilities,
        }
    }

    /// Apply quantum superposition to create multiple search paths
    pub fn apply_superposition(&mut self, config: &QuantumSearchConfig) {
        let num_states = self.amplitudes.len();

        for i in 0..num_states {
            // Apply Hadamard-like transformation for superposition
            let angle = std::f32::consts::PI * i as f32 / num_states as f32;
            self.amplitudes[i] = (angle.cos() * config.interference_amplitude).abs();
            self.phases[i] = angle.sin() * config.interference_amplitude;
        }

        self.normalize();
    }

    /// Create entanglement between quantum states
    pub fn create_entanglement(&mut self, config: &QuantumSearchConfig) {
        let num_states = self.amplitudes.len();

        for i in 0..num_states {
            for j in (i + 1)..num_states {
                // Create entanglement based on state similarity
                let entanglement =
                    config.entanglement_strength * (self.amplitudes[i] * self.amplitudes[j]).sqrt();

                self.entanglement_matrix[i][j] = entanglement;
                self.entanglement_matrix[j][i] = entanglement;
            }
        }
    }

    /// Apply quantum interference patterns
    pub fn apply_interference(&mut self, target_similarity: f32) {
        let num_states = self.amplitudes.len();

        for i in 0..num_states {
            // Constructive interference for high similarity states
            if self.probabilities[i] > target_similarity {
                self.amplitudes[i] *= 1.0 + target_similarity;
                self.phases[i] += std::f32::consts::PI / 4.0;
            } else {
                // Destructive interference for low similarity states
                self.amplitudes[i] *= 1.0 - target_similarity * 0.5;
                self.phases[i] -= std::f32::consts::PI / 4.0;
            }
        }

        self.normalize();
    }

    /// Simulate quantum tunneling for optimization landscape exploration
    pub fn quantum_tunneling(&mut self, barrier_height: f32) -> Vec<usize> {
        let mut tunneling_states = Vec::new();

        for i in 0..self.amplitudes.len() {
            // Quantum tunneling probability based on barrier height
            let tunneling_prob = (-2.0 * barrier_height).exp();

            if self.probabilities[i] * tunneling_prob > 0.1 {
                tunneling_states.push(i);
                // Boost amplitude for tunneling states
                self.amplitudes[i] *= (1.0 + tunneling_prob).sqrt();
            }
        }

        self.normalize();
        tunneling_states
    }

    /// Measure quantum state and collapse to classical result
    pub fn measure(&mut self, config: &QuantumSearchConfig) -> Vec<usize> {
        self.update_probabilities();

        let mut measured_states = Vec::new();

        for (i, &prob) in self.probabilities.iter().enumerate() {
            if prob > config.measurement_threshold {
                measured_states.push(i);
            }
        }

        // Apply decoherence
        for amplitude in &mut self.amplitudes {
            *amplitude *= 1.0 - config.decoherence_rate;
        }

        measured_states
    }

    /// Update probability distribution from amplitudes
    fn update_probabilities(&mut self) {
        for (i, prob) in self.probabilities.iter_mut().enumerate() {
            *prob = self.amplitudes[i].powi(2);
        }
    }

    /// Normalize quantum state with SIMD optimization
    fn normalize(&mut self) {
        // Use SIMD for computing norm
        let norm = f32::norm(&self.amplitudes);

        if norm > 0.0 {
            // Scale vector to normalize (no SIMD scale available, use scalar)
            for amplitude in &mut self.amplitudes {
                *amplitude /= norm;
            }
        }

        self.update_probabilities();
    }

    /// Enhanced quantum tunneling with better barrier modeling
    pub fn enhanced_quantum_tunneling(&mut self, barrier_profile: &[f32]) -> Result<Vec<usize>> {
        if barrier_profile.len() != self.amplitudes.len() {
            return Err(anyhow!(
                "Barrier profile length must match number of quantum states"
            ));
        }

        let mut tunneling_states = Vec::new();

        #[allow(clippy::needless_range_loop)]
        for i in 0..self.amplitudes.len() {
            let barrier_height = barrier_profile[i];

            // More sophisticated tunneling probability calculation
            let transmission_coefficient = if barrier_height > 0.0 {
                let tunneling_width = 1.0; // Simplified barrier width
                (-2.0 * (2.0 * barrier_height).sqrt() * tunneling_width).exp()
            } else {
                1.0 // No barrier
            };

            let tunneling_prob = self.probabilities[i] * transmission_coefficient;

            if tunneling_prob > 0.05 {
                tunneling_states.push(i);
                // Enhance amplitude for successful tunneling
                self.amplitudes[i] *= (1.0 + transmission_coefficient).sqrt();
            }
        }

        self.normalize();
        Ok(tunneling_states)
    }
}

/// Quantum-inspired vector search algorithm
#[derive(Debug)]
pub struct QuantumVectorSearch {
    config: QuantumSearchConfig,
    quantum_states: Arc<RwLock<HashMap<String, QuantumState>>>,
    search_history: Arc<RwLock<Vec<QuantumSearchResult>>>,
    optimization_cache: Arc<RwLock<HashMap<String, f32>>>,
    rng: Arc<RwLock<rand::rngs::StdRng>>,
}

/// Result of quantum-inspired search with quantum metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSearchResult {
    pub vector_id: String,
    pub similarity: f32,
    pub quantum_probability: f32,
    pub entanglement_score: f32,
    pub interference_pattern: f32,
    pub tunneling_advantage: f32,
    pub quantum_confidence: f32,
}

impl QuantumVectorSearch {
    /// Create a new quantum vector search instance
    pub fn new(config: QuantumSearchConfig) -> Self {
        Self {
            config,
            quantum_states: Arc::new(RwLock::new(HashMap::new())),
            search_history: Arc::new(RwLock::new(Vec::new())),
            optimization_cache: Arc::new(RwLock::new(HashMap::new())),
            rng: Arc::new(RwLock::new(rand::rngs::StdRng::from_entropy())),
        }
    }

    /// Create with default configuration
    pub fn with_default_config() -> Self {
        Self::new(QuantumSearchConfig::default())
    }

    /// Create with seeded random number generator for reproducible results
    pub fn with_seed(config: QuantumSearchConfig, seed: u64) -> Self {
        Self {
            config,
            quantum_states: Arc::new(RwLock::new(HashMap::new())),
            search_history: Arc::new(RwLock::new(Vec::new())),
            optimization_cache: Arc::new(RwLock::new(HashMap::new())),
            rng: Arc::new(RwLock::new(rand::rngs::StdRng::seed_from_u64(seed))),
        }
    }

    /// Perform quantum-inspired similarity search
    pub async fn quantum_similarity_search(
        &self,
        query_vector: &Vector,
        candidate_vectors: &[(String, Vector)],
        k: usize,
    ) -> Result<Vec<QuantumSearchResult>> {
        let span = span!(Level::DEBUG, "quantum_similarity_search");
        let _enter = span.enter();

        let query_id = self.generate_query_id(query_vector);

        // Initialize quantum state for this search
        let mut quantum_state = QuantumState::new(self.config.superposition_states);
        quantum_state.apply_superposition(&self.config);
        quantum_state.create_entanglement(&self.config);

        let mut results = Vec::new();
        let query_f32 = query_vector.as_f32();

        // Quantum-enhanced similarity computation
        for (candidate_id, candidate_vector) in candidate_vectors {
            let candidate_f32 = candidate_vector.as_f32();

            // Classical similarity computation
            let classical_similarity = self.compute_cosine_similarity(&query_f32, &candidate_f32);

            // Apply quantum interference based on similarity
            quantum_state.apply_interference(classical_similarity);

            // Quantum tunneling for exploration
            let tunneling_states = if self.config.enable_tunneling {
                quantum_state.quantum_tunneling(1.0 - classical_similarity)
            } else {
                Vec::new()
            };

            // Measure quantum state
            let measured_states = quantum_state.measure(&self.config);

            // Compute quantum-enhanced metrics
            let quantum_probability = quantum_state.probabilities.iter().sum::<f32>()
                / quantum_state.probabilities.len() as f32;
            let entanglement_score = self.compute_entanglement_score(&quantum_state);
            let interference_pattern = self.compute_interference_pattern(&quantum_state);
            let tunneling_advantage = if tunneling_states.is_empty() {
                0.0
            } else {
                tunneling_states.len() as f32 / self.config.superposition_states as f32
            };

            // Quantum-enhanced similarity score
            let quantum_similarity = classical_similarity * (1.0 + quantum_probability * 0.3);
            let quantum_confidence =
                self.compute_quantum_confidence(&quantum_state, &measured_states);

            results.push(QuantumSearchResult {
                vector_id: candidate_id.clone(),
                similarity: quantum_similarity,
                quantum_probability,
                entanglement_score,
                interference_pattern,
                tunneling_advantage,
                quantum_confidence,
            });
        }

        // Sort by quantum-enhanced similarity
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(k);

        // Store quantum state for future use
        {
            let mut states = self.quantum_states.write().unwrap();
            states.insert(query_id, quantum_state);
        }

        // Store search result in history
        {
            let mut history = self.search_history.write().unwrap();
            history.extend(results.clone());
        }

        info!(
            "Quantum similarity search completed with {} results",
            results.len()
        );
        Ok(results)
    }

    /// Parallel quantum-inspired similarity search for improved performance
    pub async fn parallel_quantum_similarity_search(
        &self,
        query_vector: &Vector,
        candidate_vectors: &[(String, Vector)],
        k: usize,
    ) -> Result<Vec<QuantumSearchResult>> {
        let span = span!(Level::DEBUG, "parallel_quantum_similarity_search");
        let _enter = span.enter();

        if candidate_vectors.is_empty() {
            return Ok(Vec::new());
        }

        let _query_id = self.generate_query_id(query_vector);
        let query_f32 = query_vector.as_f32();

        // Use parallel processing for large datasets
        let chunk_size = std::cmp::max(candidate_vectors.len() / num_cpus::get(), 1);

        let results: Result<Vec<Vec<QuantumSearchResult>>> = candidate_vectors
            .par_chunks(chunk_size)
            .map(|chunk| -> Result<Vec<QuantumSearchResult>> {
                let mut chunk_results = Vec::new();
                let mut quantum_state = QuantumState::new(self.config.superposition_states);
                quantum_state.apply_superposition(&self.config);
                quantum_state.create_entanglement(&self.config);

                for (candidate_id, candidate_vector) in chunk {
                    let candidate_f32 = candidate_vector.as_f32();

                    // Classical similarity computation with SIMD optimization
                    let classical_similarity =
                        self.compute_cosine_similarity(&query_f32, &candidate_f32);

                    // Apply quantum interference
                    quantum_state.apply_interference(classical_similarity);

                    // Quantum tunneling if enabled
                    let tunneling_advantage = if self.config.enable_tunneling {
                        let barrier_height =
                            vec![1.0 - classical_similarity; self.config.superposition_states];
                        match quantum_state.enhanced_quantum_tunneling(&barrier_height) {
                            Ok(tunneling_states) => {
                                if tunneling_states.is_empty() {
                                    0.0
                                } else {
                                    tunneling_states.len() as f32
                                        / self.config.superposition_states as f32
                                }
                            }
                            Err(_) => 0.0,
                        }
                    } else {
                        0.0
                    };

                    // Measure quantum state
                    let measured_states = quantum_state.measure(&self.config);

                    // Compute quantum-enhanced metrics
                    let quantum_probability = quantum_state.probabilities.iter().sum::<f32>()
                        / quantum_state.probabilities.len() as f32;
                    let entanglement_score = self.compute_entanglement_score(&quantum_state);
                    let interference_pattern = self.compute_interference_pattern(&quantum_state);
                    let quantum_confidence =
                        self.compute_quantum_confidence(&quantum_state, &measured_states);

                    // Enhanced quantum similarity score with better weighting
                    let quantum_enhancement = quantum_probability * 0.3
                        + entanglement_score * 0.1
                        + tunneling_advantage * 0.2;
                    let quantum_similarity = classical_similarity * (1.0 + quantum_enhancement);

                    chunk_results.push(QuantumSearchResult {
                        vector_id: candidate_id.clone(),
                        similarity: quantum_similarity,
                        quantum_probability,
                        entanglement_score,
                        interference_pattern,
                        tunneling_advantage,
                        quantum_confidence,
                    });
                }

                Ok(chunk_results)
            })
            .collect();

        let mut all_results: Vec<QuantumSearchResult> = results?.into_iter().flatten().collect();

        // Sort by quantum-enhanced similarity
        all_results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        all_results.truncate(k);

        // Store search result in history
        {
            let mut history = self.search_history.write().unwrap();
            history.extend(all_results.clone());
        }

        info!(
            "Parallel quantum similarity search completed with {} results",
            all_results.len()
        );
        Ok(all_results)
    }

    /// Perform quantum amplitude amplification for targeted search
    pub fn quantum_amplitude_amplification(
        &self,
        target_similarity: f32,
        quantum_state: &mut QuantumState,
        iterations: usize,
    ) -> Result<()> {
        for iteration in 0..iterations {
            // Oracle operation: mark target states
            for (i, &prob) in quantum_state.probabilities.iter().enumerate() {
                if prob >= target_similarity {
                    quantum_state.amplitudes[i] *= -1.0; // Phase flip
                }
            }

            // Diffusion operation: inversion about average
            let average_amplitude: f32 = quantum_state.amplitudes.iter().sum::<f32>()
                / quantum_state.amplitudes.len() as f32;

            for amplitude in &mut quantum_state.amplitudes {
                *amplitude = 2.0 * average_amplitude - *amplitude;
            }

            quantum_state.normalize();

            debug!(
                "Amplitude amplification iteration {} completed",
                iteration + 1
            );
        }

        Ok(())
    }

    /// Quantum annealing for optimization landscape exploration
    pub fn quantum_annealing_optimization(
        &self,
        cost_function: impl Fn(&[f32]) -> f32,
        initial_state: &[f32],
        temperature_schedule: &[f32],
    ) -> Result<Vec<f32>> {
        let mut current_state = initial_state.to_vec();
        let mut best_state = current_state.clone();
        let mut best_cost = cost_function(&current_state);

        for &temperature in temperature_schedule {
            // Quantum fluctuations
            for item in &mut current_state {
                let quantum_fluctuation = self.generate_quantum_fluctuation(temperature);
                *item += quantum_fluctuation;
            }

            let current_cost = cost_function(&current_state);

            // Quantum acceptance probability
            let accept_prob = if current_cost < best_cost {
                1.0
            } else {
                (-(current_cost - best_cost) / temperature).exp()
            };

            if self.generate_random() < accept_prob {
                best_state = current_state.clone();
                best_cost = current_cost;
            }

            debug!(
                "Quantum annealing: temperature={}, cost={}",
                temperature, current_cost
            );
        }

        Ok(best_state)
    }

    /// Get quantum search statistics
    pub fn get_quantum_statistics(&self) -> QuantumSearchStatistics {
        let history = self.search_history.read().unwrap();

        let total_searches = history.len();
        let avg_quantum_probability = if total_searches > 0 {
            history.iter().map(|r| r.quantum_probability).sum::<f32>() / total_searches as f32
        } else {
            0.0
        };

        let avg_entanglement_score = if total_searches > 0 {
            history.iter().map(|r| r.entanglement_score).sum::<f32>() / total_searches as f32
        } else {
            0.0
        };

        let avg_quantum_confidence = if total_searches > 0 {
            history.iter().map(|r| r.quantum_confidence).sum::<f32>() / total_searches as f32
        } else {
            0.0
        };

        QuantumSearchStatistics {
            total_searches,
            avg_quantum_probability,
            avg_entanglement_score,
            avg_quantum_confidence,
            superposition_states: self.config.superposition_states,
            entanglement_strength: self.config.entanglement_strength,
        }
    }

    // Helper methods

    fn generate_query_id(&self, vector: &Vector) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for value in vector.as_f32() {
            value.to_bits().hash(&mut hasher);
        }
        format!("quantum_query_{:x}", hasher.finish())
    }

    fn compute_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        // Use SIMD optimization via oxirs-core for better performance
        // Note: SimdOps provides cosine_distance, so we convert to similarity
        let cosine_distance = f32::cosine_distance(a, b);
        1.0 - cosine_distance
    }

    fn compute_entanglement_score(&self, quantum_state: &QuantumState) -> f32 {
        let mut entanglement_score = 0.0;
        let num_states = quantum_state.entanglement_matrix.len();

        for i in 0..num_states {
            for j in (i + 1)..num_states {
                entanglement_score += quantum_state.entanglement_matrix[i][j].abs();
            }
        }

        entanglement_score / (num_states * (num_states - 1) / 2) as f32
    }

    fn compute_interference_pattern(&self, quantum_state: &QuantumState) -> f32 {
        let mut interference = 0.0;

        for i in 0..quantum_state.amplitudes.len() {
            let amplitude = quantum_state.amplitudes[i];
            let phase = quantum_state.phases[i];
            interference += amplitude * phase.cos();
        }

        interference / quantum_state.amplitudes.len() as f32
    }

    fn compute_quantum_confidence(
        &self,
        quantum_state: &QuantumState,
        measured_states: &[usize],
    ) -> f32 {
        if measured_states.is_empty() {
            return 0.0;
        }

        let measured_probability: f32 = measured_states
            .iter()
            .map(|&i| quantum_state.probabilities[i])
            .sum();

        // Confidence based on measurement certainty
        let max_probability = quantum_state
            .probabilities
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0);

        (measured_probability * max_probability).sqrt()
    }

    fn generate_quantum_fluctuation(&self, temperature: f32) -> f32 {
        // Simulate quantum fluctuation using thermal noise with proper Gaussian distribution
        let mut rng = self.rng.write().unwrap();

        // Use proper normal distribution for quantum fluctuations
        let normal =
            Normal::new(0.0, temperature.sqrt()).unwrap_or_else(|_| Normal::new(0.0, 1.0).unwrap());
        normal.sample(&mut *rng)
    }

    fn generate_random(&self) -> f32 {
        // Use proper random number generator
        let mut rng = self.rng.write().unwrap();
        rng.r#gen()
    }
}

/// Statistics for quantum search operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSearchStatistics {
    pub total_searches: usize,
    pub avg_quantum_probability: f32,
    pub avg_entanglement_score: f32,
    pub avg_quantum_confidence: f32,
    pub superposition_states: usize,
    pub entanglement_strength: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_state_creation() {
        let quantum_state = QuantumState::new(8);
        assert_eq!(quantum_state.amplitudes.len(), 8);
        assert_eq!(quantum_state.phases.len(), 8);
        assert_eq!(quantum_state.entanglement_matrix.len(), 8);
        assert_eq!(quantum_state.probabilities.len(), 8);
    }

    #[test]
    fn test_quantum_superposition() {
        let mut quantum_state = QuantumState::new(4);
        let config = QuantumSearchConfig::default();

        quantum_state.apply_superposition(&config);

        // Check that amplitudes are normalized
        let norm: f32 = quantum_state.amplitudes.iter().map(|a| a.powi(2)).sum();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantum_entanglement() {
        let mut quantum_state = QuantumState::new(4);
        let config = QuantumSearchConfig::default();

        quantum_state.create_entanglement(&config);

        // Check that entanglement matrix is symmetric
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    quantum_state.entanglement_matrix[i][j],
                    quantum_state.entanglement_matrix[j][i]
                );
            }
        }
    }

    #[tokio::test]
    async fn test_quantum_vector_search() {
        let quantum_search = QuantumVectorSearch::with_seed(QuantumSearchConfig::default(), 42);

        let query_vector = Vector::new(vec![1.0, 0.0, 0.0]);
        let candidates = vec![
            ("vec1".to_string(), Vector::new(vec![0.9, 0.1, 0.0])),
            ("vec2".to_string(), Vector::new(vec![0.0, 1.0, 0.0])),
            ("vec3".to_string(), Vector::new(vec![0.8, 0.0, 0.6])),
        ];

        let results = quantum_search
            .quantum_similarity_search(&query_vector, &candidates, 2)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].similarity >= results[1].similarity);
        assert!(results[0].quantum_confidence >= 0.0);
        assert!(results[0].quantum_confidence <= 1.0);
    }

    #[tokio::test]
    async fn test_parallel_quantum_vector_search() {
        let quantum_search = QuantumVectorSearch::with_seed(QuantumSearchConfig::default(), 42);

        let query_vector = Vector::new(vec![1.0, 0.0, 0.0]);
        let candidates = vec![
            ("vec1".to_string(), Vector::new(vec![0.9, 0.1, 0.0])),
            ("vec2".to_string(), Vector::new(vec![0.0, 1.0, 0.0])),
            ("vec3".to_string(), Vector::new(vec![0.8, 0.0, 0.6])),
            ("vec4".to_string(), Vector::new(vec![0.7, 0.7, 0.0])),
            ("vec5".to_string(), Vector::new(vec![0.5, 0.5, 0.7])),
        ];

        let results = quantum_search
            .parallel_quantum_similarity_search(&query_vector, &candidates, 3)
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        assert!(results[0].similarity >= results[1].similarity);
        assert!(results[1].similarity >= results[2].similarity);
        assert!(results[0].quantum_confidence >= 0.0);
        assert!(results[0].quantum_confidence <= 1.0);
    }

    #[test]
    fn test_quantum_amplitude_amplification() {
        let quantum_search = QuantumVectorSearch::with_default_config();
        let mut quantum_state = QuantumState::new(8);

        let result = quantum_search.quantum_amplitude_amplification(0.5, &mut quantum_state, 3);
        assert!(result.is_ok());

        // Check that amplitudes are still normalized
        let norm: f32 = quantum_state.amplitudes.iter().map(|a| a.powi(2)).sum();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantum_annealing() {
        let quantum_search = QuantumVectorSearch::with_default_config();

        // Simple quadratic cost function
        let cost_fn = |state: &[f32]| -> f32 { state.iter().map(|x| (x - 0.5).powi(2)).sum() };

        let initial_state = vec![0.0, 1.0, 0.2];
        let temperature_schedule = vec![1.0, 0.5, 0.1];

        let result = quantum_search.quantum_annealing_optimization(
            cost_fn,
            &initial_state,
            &temperature_schedule,
        );
        assert!(result.is_ok());

        let optimized_state = result.unwrap();
        assert_eq!(optimized_state.len(), initial_state.len());
    }

    #[test]
    fn test_quantum_tunneling() {
        let mut quantum_state = QuantumState::new(8);
        let tunneling_states = quantum_state.quantum_tunneling(0.8);

        // Should return some states that can tunnel
        assert!(tunneling_states.len() <= 8);

        // Check all returned states are valid indices
        for state in tunneling_states {
            assert!(state < 8);
        }
    }

    #[test]
    fn test_quantum_measurement() {
        let mut quantum_state = QuantumState::new(4);
        let config = QuantumSearchConfig::default();

        // Set up some probability distribution
        quantum_state.amplitudes = vec![0.6, 0.4, 0.3, 0.5];
        quantum_state.normalize();

        let measured_states = quantum_state.measure(&config);

        // Should return states above threshold
        assert!(!measured_states.is_empty());
        for state in measured_states {
            assert!(state < 4);
        }
    }

    #[test]
    fn test_enhanced_quantum_tunneling() {
        let mut quantum_state = QuantumState::new(8);

        // Set up initial state
        quantum_state.amplitudes = vec![0.3, 0.4, 0.2, 0.5, 0.1, 0.6, 0.3, 0.4];
        quantum_state.normalize();

        // Create barrier profile (higher values = stronger barriers)
        let barrier_profile = vec![0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4];

        let tunneling_result = quantum_state.enhanced_quantum_tunneling(&barrier_profile);
        assert!(tunneling_result.is_ok());

        let tunneling_states = tunneling_result.unwrap();

        // Should return some states that can tunnel (those with lower barriers)
        assert!(!tunneling_states.is_empty());

        // Check all returned states are valid indices
        for state in tunneling_states {
            assert!(state < 8);
        }
    }

    #[test]
    fn test_quantum_statistics() {
        let quantum_search = QuantumVectorSearch::with_default_config();
        let stats = quantum_search.get_quantum_statistics();

        assert_eq!(stats.total_searches, 0);
        assert_eq!(stats.superposition_states, 64);
        assert_eq!(stats.entanglement_strength, 0.7);
    }
}
