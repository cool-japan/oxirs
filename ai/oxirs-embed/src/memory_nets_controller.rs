//! Controller networks for Memory-Augmented Networks (DNC and NTM).
//!
//! This module contains:
//! - DNC configuration and implementation
//! - NTM configuration and implementation
//! - Controller network (LSTM-style)
//! - Read and write heads for DNC
//! - Memory addressing sub-systems (allocation, temporal linkage, usage tracker)
//! - NTM heads with content/shift/sharpen addressing

use anyhow::{anyhow, Result};
use scirs2_core::ndarray::concatenate as ndarray_concatenate;
use scirs2_core::ndarray_ext::{s, Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// DNC Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DNCConfig {
    /// Number of memory slots
    pub memory_size: usize,
    /// Size of each memory slot
    pub memory_width: usize,
    /// Number of read heads
    pub num_read_heads: usize,
    /// Controller network size
    pub controller_size: usize,
    /// Output size
    pub output_size: usize,
    /// Learning rate for memory operations
    pub memory_learning_rate: f32,
    /// Memory decay factor
    pub memory_decay: f32,
}

impl Default for DNCConfig {
    fn default() -> Self {
        Self {
            memory_size: 256,
            memory_width: 64,
            num_read_heads: 4,
            controller_size: 512,
            output_size: 256,
            memory_learning_rate: 0.001,
            memory_decay: 0.95,
        }
    }
}

/// Controller network for DNC (LSTM-style)
pub struct ControllerNetwork {
    /// Input to hidden weights
    pub(crate) w_ih: Array2<f32>,
    /// Hidden to hidden weights
    pub(crate) w_hh: Array2<f32>,
    /// Hidden to output weights
    pub(crate) w_ho: Array2<f32>,
    /// Bias vectors
    pub(crate) bias_h: Array1<f32>,
    pub(crate) bias_o: Array1<f32>,
    /// Hidden state
    pub(crate) hidden_state: Array1<f32>,
    /// Cell state (for LSTM)
    pub(crate) cell_state: Array1<f32>,
}

impl ControllerNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        use scirs2_core::random::Random;
        let mut rng = Random::default();

        let w_ih =
            Array2::from_shape_fn((hidden_size, input_size), |_| rng.random_range(-0.1..0.1));
        let w_hh =
            Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.random_range(-0.1..0.1));
        let w_ho =
            Array2::from_shape_fn((output_size, hidden_size), |_| rng.random_range(-0.1..0.1));
        let bias_h = Array1::zeros(hidden_size);
        let bias_o = Array1::zeros(output_size);
        let hidden_state = Array1::zeros(hidden_size);
        let cell_state = Array1::zeros(hidden_size);

        Self {
            w_ih,
            w_hh,
            w_ho,
            bias_h,
            bias_o,
            hidden_state,
            cell_state,
        }
    }

    /// Forward pass through controller (LSTM-style computation)
    pub fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let input_gate = self
            .sigmoid(&(&self.w_ih.dot(input) + &self.w_hh.dot(&self.hidden_state) + &self.bias_h));
        let forget_gate = self
            .sigmoid(&(&self.w_ih.dot(input) + &self.w_hh.dot(&self.hidden_state) + &self.bias_h));
        let cell_gate =
            self.tanh(&(&self.w_ih.dot(input) + &self.w_hh.dot(&self.hidden_state) + &self.bias_h));
        let output_gate = self
            .sigmoid(&(&self.w_ih.dot(input) + &self.w_hh.dot(&self.hidden_state) + &self.bias_h));

        self.cell_state = &forget_gate * &self.cell_state + &input_gate * &cell_gate;
        self.hidden_state = &output_gate * &self.tanh(&self.cell_state);

        self.w_ho.dot(&self.hidden_state) + &self.bias_o
    }

    fn sigmoid(&self, x: &Array1<f32>) -> Array1<f32> {
        x.map(|&v| 1.0 / (1.0 + (-v).exp()))
    }

    fn tanh(&self, x: &Array1<f32>) -> Array1<f32> {
        x.map(|&v| v.tanh())
    }
}

/// Read head for DNC
pub struct ReadHead {
    /// Key vector for content-based addressing
    pub(crate) key: Array1<f32>,
    /// Key strength
    pub(crate) key_strength: f32,
    /// Free gates for memory deallocation
    pub(crate) free_gates: Array1<f32>,
    /// Read modes (backward, forward, content lookup)
    pub(crate) read_modes: Array1<f32>,
}

impl ReadHead {
    pub fn new(memory_width: usize) -> Self {
        Self {
            key: Array1::zeros(memory_width),
            key_strength: 1.0,
            free_gates: Array1::zeros(memory_width),
            read_modes: Array1::from_vec(vec![1.0, 0.0, 0.0]),
        }
    }

    /// Generate read weighting using content-based + temporal addressing
    pub fn generate_weighting(
        &self,
        memory: &Array2<f32>,
        link_matrix: &Array2<f32>,
        prev_read_weighting: &Array1<f32>,
    ) -> Array1<f32> {
        let content_weighting = self.content_lookup(memory);
        let forward_weighting = link_matrix.dot(prev_read_weighting);
        let backward_weighting = link_matrix.t().dot(prev_read_weighting);

        let combined_weighting = self.read_modes[0] * &backward_weighting
            + self.read_modes[1] * &content_weighting
            + self.read_modes[2] * &forward_weighting;

        let sum = combined_weighting.sum();
        if sum > 0.0 {
            combined_weighting / sum
        } else {
            Array1::zeros(memory.nrows())
        }
    }

    fn content_lookup(&self, memory: &Array2<f32>) -> Array1<f32> {
        let mut similarities = Array1::zeros(memory.nrows());
        for (i, memory_row) in memory.axis_iter(Axis(0)).enumerate() {
            similarities[i] = cosine_similarity(&self.key, &memory_row.to_owned());
        }
        let scaled = similarities.map(|&x| (x * self.key_strength).exp());
        let sum = scaled.sum();
        if sum > 0.0 {
            scaled / sum
        } else {
            Array1::zeros(memory.nrows())
        }
    }
}

/// Write head for DNC
pub struct WriteHead {
    pub(crate) key: Array1<f32>,
    pub(crate) key_strength: f32,
    pub(crate) erase_vector: Array1<f32>,
    pub(crate) write_vector: Array1<f32>,
    pub(crate) allocation_gate: f32,
    pub(crate) write_gate: f32,
}

impl WriteHead {
    pub fn new(memory_width: usize) -> Self {
        Self {
            key: Array1::zeros(memory_width),
            key_strength: 1.0,
            erase_vector: Array1::zeros(memory_width),
            write_vector: Array1::zeros(memory_width),
            allocation_gate: 0.0,
            write_gate: 1.0,
        }
    }

    /// Generate write weighting combining content-based and allocation
    pub fn generate_weighting(
        &self,
        memory: &Array2<f32>,
        usage_vector: &Array1<f32>,
    ) -> Array1<f32> {
        let content_weighting = self.content_lookup(memory);
        let allocation_weighting = self.allocation_lookup(usage_vector);

        self.write_gate
            * (self.allocation_gate * allocation_weighting
                + (1.0 - self.allocation_gate) * content_weighting)
    }

    fn content_lookup(&self, memory: &Array2<f32>) -> Array1<f32> {
        let mut similarities = Array1::zeros(memory.nrows());
        for (i, memory_row) in memory.axis_iter(Axis(0)).enumerate() {
            similarities[i] = cosine_similarity(&self.key, &memory_row.to_owned());
        }
        let scaled = similarities.map(|&x| (x * self.key_strength).exp());
        let sum = scaled.sum();
        if sum > 0.0 {
            scaled / sum
        } else {
            Array1::zeros(memory.nrows())
        }
    }

    fn allocation_lookup(&self, usage_vector: &Array1<f32>) -> Array1<f32> {
        let mut indices: Vec<usize> = (0..usage_vector.len()).collect();
        indices.sort_by(|&a, &b| {
            usage_vector[a]
                .partial_cmp(&usage_vector[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut allocation = Array1::zeros(usage_vector.len());
        for (rank, &idx) in indices.iter().enumerate() {
            allocation[idx] = 1.0 / (rank as f32 + 1.0);
        }
        let sum = allocation.sum();
        if sum > 0.0 {
            allocation / sum
        } else {
            Array1::zeros(usage_vector.len())
        }
    }

    /// Erase-then-write to memory
    pub fn write_to_memory(&self, memory: &mut Array2<f32>, weighting: &Array1<f32>) {
        for i in 0..memory.nrows() {
            for j in 0..memory.ncols() {
                memory[[i, j]] *= 1.0 - weighting[i] * self.erase_vector[j];
            }
        }
        for i in 0..memory.nrows() {
            for j in 0..memory.ncols() {
                memory[[i, j]] += weighting[i] * self.write_vector[j];
            }
        }
    }
}

/// Usage tracking for memory allocation
pub struct UsageTracker {
    pub(crate) usage: Array1<f32>,
    pub(crate) memory_size: usize,
}

impl UsageTracker {
    pub fn new(memory_size: usize) -> Self {
        Self {
            usage: Array1::zeros(memory_size),
            memory_size,
        }
    }

    pub fn update(&mut self, write_weighting: &Array1<f32>, free_gates: &Array1<f32>) {
        for i in 0..self.memory_size {
            self.usage[i] = (self.usage[i] + write_weighting[i] - self.usage[i] * free_gates[i])
                .clamp(0.0, 1.0);
        }
    }

    pub fn get_allocation_weighting(&self, _allocation_gate: f32) -> Array1<f32> {
        let mut sorted_indices: Vec<usize> = (0..self.memory_size).collect();
        sorted_indices.sort_by(|&a, &b| {
            self.usage[a]
                .partial_cmp(&self.usage[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut weights = Array1::zeros(self.memory_size);
        for (rank, &idx) in sorted_indices.iter().enumerate() {
            weights[idx] = 1.0 / (rank as f32 + 1.0);
        }
        let sum = weights.sum();
        if sum > 0.0 {
            weights / sum
        } else {
            Array1::zeros(self.memory_size)
        }
    }
}

/// Allocation mechanism for finding free memory
pub struct AllocationMechanism {
    pub(crate) usage_tracker: UsageTracker,
}

impl AllocationMechanism {
    pub fn new(memory_size: usize) -> Self {
        Self {
            usage_tracker: UsageTracker::new(memory_size),
        }
    }

    pub fn allocate(&mut self, allocation_gate: f32) -> Array1<f32> {
        self.usage_tracker.get_allocation_weighting(allocation_gate)
    }

    pub fn update_usage(&mut self, write_weighting: &Array1<f32>, free_gates: &Array1<f32>) {
        self.usage_tracker.update(write_weighting, free_gates);
    }
}

/// Temporal linkage for sequential memory access
pub struct TemporalLinkage {
    pub(crate) link_matrix: Array2<f32>,
    pub(crate) precedence_weighting: Array1<f32>,
}

impl TemporalLinkage {
    pub fn new(memory_size: usize) -> Self {
        Self {
            link_matrix: Array2::zeros((memory_size, memory_size)),
            precedence_weighting: Array1::zeros(memory_size),
        }
    }

    pub fn update(&mut self, write_weighting: &Array1<f32>) {
        let sum = write_weighting.sum();
        if sum > 0.0 {
            self.precedence_weighting = (1.0 - sum) * &self.precedence_weighting + write_weighting;
        }
        for i in 0..self.link_matrix.nrows() {
            for j in 0..self.link_matrix.ncols() {
                if i != j {
                    self.link_matrix[[i, j]] = (1.0 - write_weighting[i] - write_weighting[j])
                        * self.link_matrix[[i, j]]
                        + write_weighting[i] * self.precedence_weighting[j];
                }
            }
        }
    }

    pub fn get_link_matrix(&self) -> &Array2<f32> {
        &self.link_matrix
    }
}

/// Memory addressing system
pub struct MemoryAddressing {
    pub(crate) allocation_mechanism: AllocationMechanism,
    pub(crate) temporal_linkage: TemporalLinkage,
}

/// Differentiable Neural Computer implementation
pub struct DifferentiableNeuralComputer {
    pub(crate) config: DNCConfig,
    pub(crate) controller: ControllerNetwork,
    pub(crate) memory_matrix: Array2<f32>,
    pub(crate) read_heads: Vec<ReadHead>,
    pub(crate) write_head: WriteHead,
    pub(crate) memory_addressing: MemoryAddressing,
    pub(crate) usage_vector: Array1<f32>,
    pub(crate) precedence_weights: Array1<f32>,
    pub(crate) link_matrix: Array2<f32>,
    pub(crate) read_weightings: Array2<f32>,
    pub(crate) write_weighting: Array1<f32>,
}

impl DifferentiableNeuralComputer {
    /// Create new DNC
    pub fn new(config: DNCConfig) -> Self {
        let memory_matrix = Array2::zeros((config.memory_size, config.memory_width));
        let usage_vector = Array1::zeros(config.memory_size);
        let precedence_weights = Array1::zeros(config.memory_size);
        let link_matrix = Array2::zeros((config.memory_size, config.memory_size));
        let read_weightings = Array2::zeros((config.num_read_heads, config.memory_size));
        let write_weighting = Array1::zeros(config.memory_size);

        let controller = ControllerNetwork::new(
            config.memory_width + config.num_read_heads * config.memory_width,
            config.controller_size,
            config.output_size
                + config.memory_width * (config.num_read_heads + 1)
                + 3 * config.num_read_heads
                + 5,
        );

        let read_heads = (0..config.num_read_heads)
            .map(|_| ReadHead::new(config.memory_width))
            .collect();

        let write_head = WriteHead::new(config.memory_width);

        let memory_addressing = MemoryAddressing {
            allocation_mechanism: AllocationMechanism::new(config.memory_size),
            temporal_linkage: TemporalLinkage::new(config.memory_size),
        };

        Self {
            config,
            controller,
            memory_matrix,
            read_heads,
            write_head,
            memory_addressing,
            usage_vector,
            precedence_weights,
            link_matrix,
            read_weightings,
            write_weighting,
        }
    }

    /// Forward pass through DNC
    pub fn forward(&mut self, input: &Array1<f32>) -> Result<Array1<f32>> {
        let mut read_vectors = Vec::new();
        for (i, read_head) in self.read_heads.iter().enumerate() {
            let read_weighting = read_head.generate_weighting(
                &self.memory_matrix,
                &self.link_matrix,
                &self.read_weightings.row(i).to_owned(),
            );
            let read_vector = self.memory_matrix.t().dot(&read_weighting);
            read_vectors.push(read_vector);
        }

        let mut controller_input = input.clone();
        for read_vector in &read_vectors {
            let views: &[_] = &[controller_input.view(), read_vector.view()];
            controller_input = ndarray_concatenate(Axis(0), views)
                .map_err(|e| anyhow!("concatenate failed: {}", e))?;
        }

        let controller_output = self.controller.forward(&controller_input);
        let (output, _interface_vector) = self.parse_controller_output(&controller_output)?;

        let write_weighting = self
            .write_head
            .generate_weighting(&self.memory_matrix, &self.usage_vector);
        self.write_head
            .write_to_memory(&mut self.memory_matrix, &write_weighting);

        let free_gates = Array1::ones(self.config.memory_size);
        self.memory_addressing
            .allocation_mechanism
            .update_usage(&write_weighting, &free_gates);
        self.memory_addressing
            .temporal_linkage
            .update(&write_weighting);

        self.write_weighting = write_weighting;
        self.link_matrix = self
            .memory_addressing
            .temporal_linkage
            .get_link_matrix()
            .clone();

        Ok(output)
    }

    fn parse_controller_output(&self, output: &Array1<f32>) -> Result<(Array1<f32>, Array1<f32>)> {
        if output.len() < self.config.output_size {
            return Err(anyhow!("Controller output too short"));
        }
        let network_output = output.slice(s![..self.config.output_size]).to_owned();
        let interface_vector = output.slice(s![self.config.output_size..]).to_owned();
        Ok((network_output, interface_vector))
    }

    /// Reset memory state
    pub fn reset(&mut self) {
        self.memory_matrix.fill(0.0);
        self.usage_vector.fill(0.0);
        self.precedence_weights.fill(0.0);
        self.link_matrix.fill(0.0);
        self.read_weightings.fill(0.0);
        self.write_weighting.fill(0.0);
    }

    /// Get memory utilization
    pub fn get_memory_utilization(&self) -> f32 {
        self.usage_vector.sum() / self.usage_vector.len() as f32
    }
}

/// Neural Turing Machine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NTMConfig {
    pub memory_size: usize,
    pub memory_width: usize,
    pub num_heads: usize,
    pub controller_size: usize,
    pub shift_range: usize,
}

impl Default for NTMConfig {
    fn default() -> Self {
        Self {
            memory_size: 128,
            memory_width: 32,
            num_heads: 2,
            controller_size: 256,
            shift_range: 3,
        }
    }
}

/// NTM Head (read or write)
pub struct NTMHead {
    pub(crate) key: Array1<f32>,
    pub(crate) key_strength: f32,
    pub(crate) gate: f32,
    pub(crate) shift_weights: Array1<f32>,
    pub(crate) gamma: f32,
    pub(crate) prev_weighting: Array1<f32>,
}

impl NTMHead {
    pub fn new(memory_width: usize, memory_size: usize, shift_range: usize) -> Self {
        Self {
            key: Array1::zeros(memory_width),
            key_strength: 1.0,
            gate: 0.5,
            shift_weights: Array1::zeros(2 * shift_range + 1),
            gamma: 1.0,
            prev_weighting: Array1::zeros(memory_size),
        }
    }

    /// Generate addressing weighting via content + gate + shift + sharpen
    pub fn address(&mut self, memory: &Array2<f32>) -> Array1<f32> {
        let content_weights = self.content_addressing(memory);
        let gated_weights = self.gate * &content_weights + (1.0 - self.gate) * &self.prev_weighting;
        let shifted_weights = self.shift_addressing(&gated_weights);
        let final_weights = self.sharpen_addressing(&shifted_weights);
        self.prev_weighting = final_weights.clone();
        final_weights
    }

    fn content_addressing(&self, memory: &Array2<f32>) -> Array1<f32> {
        let mut similarities = Array1::zeros(memory.nrows());
        for (i, memory_row) in memory.axis_iter(Axis(0)).enumerate() {
            similarities[i] = cosine_similarity(&self.key, &memory_row.to_owned());
        }
        let scaled = similarities.map(|&x| (x * self.key_strength).exp());
        let sum = scaled.sum();
        if sum > 0.0 {
            scaled / sum
        } else {
            Array1::zeros(memory.nrows())
        }
    }

    fn shift_addressing(&self, weights: &Array1<f32>) -> Array1<f32> {
        let memory_size = weights.len();
        let shift_range = (self.shift_weights.len() - 1) / 2;
        let mut shifted = Array1::zeros(memory_size);

        for i in 0..memory_size {
            for (j, &shift_weight) in self.shift_weights.iter().enumerate() {
                let shift = j as i32 - shift_range as i32;
                let shifted_idx = ((i as i32 + shift) % memory_size as i32 + memory_size as i32)
                    % memory_size as i32;
                shifted[shifted_idx as usize] += weights[i] * shift_weight;
            }
        }
        shifted
    }

    fn sharpen_addressing(&self, weights: &Array1<f32>) -> Array1<f32> {
        let sharpened = weights.map(|&x| x.powf(self.gamma));
        let sum = sharpened.sum();
        if sum > 0.0 {
            sharpened / sum
        } else {
            Array1::zeros(weights.len())
        }
    }
}

/// Neural Turing Machine implementation
pub struct NeuralTuringMachine {
    pub(crate) config: NTMConfig,
    pub(crate) controller: ControllerNetwork,
    pub(crate) memory: Array2<f32>,
    pub(crate) read_heads: Vec<NTMHead>,
    pub(crate) write_heads: Vec<NTMHead>,
}

impl NeuralTuringMachine {
    pub fn new(config: NTMConfig) -> Self {
        let memory = Array2::zeros((config.memory_size, config.memory_width));
        let controller = ControllerNetwork::new(
            config.memory_width + config.num_heads * config.memory_width,
            config.controller_size,
            config.memory_width
                + config.num_heads * (config.memory_width + 3 + 2 * config.shift_range + 1),
        );

        let read_heads = (0..config.num_heads)
            .map(|_| NTMHead::new(config.memory_width, config.memory_size, config.shift_range))
            .collect();

        let write_heads = (0..config.num_heads)
            .map(|_| NTMHead::new(config.memory_width, config.memory_size, config.shift_range))
            .collect();

        Self {
            config,
            controller,
            memory,
            read_heads,
            write_heads,
        }
    }

    /// Forward pass through NTM
    pub fn forward(&mut self, input: &Array1<f32>) -> Result<Array1<f32>> {
        let mut read_vectors = Vec::new();
        for read_head in &mut self.read_heads {
            let weighting = read_head.address(&self.memory);
            let read_vector = self.memory.t().dot(&weighting);
            read_vectors.push(read_vector);
        }

        let mut controller_input = input.clone();
        for read_vector in &read_vectors {
            let views: &[_] = &[controller_input.view(), read_vector.view()];
            controller_input = ndarray_concatenate(Axis(0), views)
                .map_err(|e| anyhow!("concatenate failed: {}", e))?;
        }

        let controller_output = self.controller.forward(&controller_input);
        Ok(controller_output)
    }
}

/// Shared cosine similarity helper
pub(crate) fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot_product = a.dot(b);
    let norm_a = a.mapv(|x| x * x).sum().sqrt();
    let norm_b = b.mapv(|x| x * x).sum().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}
