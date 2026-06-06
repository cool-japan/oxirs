//! Evolutionary NAS — Evaluation
//!
//! Architecture fitness evaluation, proxy metrics, and early stopping.

use crate::evolutionary_nas_types::{
    ArchitectureCandidate, EvaluationDataset, HardwareTarget, PerformanceMetrics, ProfilingResult,
};
use anyhow::Result;
use scirs2_core::random::Random;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// ── Hardware Profiler ─────────────────────────────────────────────────────────

/// Hardware profiler for performance measurement
pub struct HardwareProfiler {
    pub target_hardware: HardwareTarget,
    pub profiling_history: Vec<ProfilingResult>,
}

// ── Fitness Evaluator ─────────────────────────────────────────────────────────

/// Fitness evaluator for architecture candidates
pub struct FitnessEvaluator {
    pub datasets: HashMap<String, EvaluationDataset>,
    pub hardware_profiler: HardwareProfiler,
    pub evaluation_cache: Arc<RwLock<HashMap<String, PerformanceMetrics>>>,
}

impl FitnessEvaluator {
    /// Evaluate the performance of a single candidate.
    ///
    /// In a real implementation this would train and evaluate the full architecture; here we use
    /// synthetic metrics that are deterministic enough for testing.
    pub async fn evaluate_candidate_performance(
        &self,
        _candidate: &ArchitectureCandidate,
    ) -> Result<PerformanceMetrics> {
        let mut random = Random::default();
        Ok(PerformanceMetrics {
            training_accuracy: random.random_range(0.7f32..0.95f32),
            validation_accuracy: random.random_range(0.65f32..0.9f32),
            test_accuracy: None,
            training_time: random.random_range(100.0f64..1000.0f64),
            inference_time_ms: random.random_range(0.1f32..10.0f32),
            memory_usage_mb: random.random_range(100.0f32..2000.0f32),
            energy_consumption: Some(random.random_range(10.0f32..100.0f32)),
            model_size: random.random_range(1_000_000usize..50_000_000usize),
            flops: random.random_range(1_000_000u64..100_000_000u64),
        })
    }
}

// ── Proxy metrics ─────────────────────────────────────────────────────────────

/// Compute a lightweight proxy accuracy score for an architecture candidate without full training.
///
/// The proxy is computed from structural features of the genome: more nodes and active connections
/// generally correlate with higher representational capacity, bounded to a plausible range.
pub fn compute_proxy_accuracy(candidate: &ArchitectureCandidate) -> f32 {
    let num_nodes = candidate.genome.nodes.len() as f32;
    let num_active = candidate
        .genome
        .connections
        .iter()
        .filter(|c| c.active)
        .count() as f32;

    // Simple heuristic: log-scale capacity score, clipped to [0.5, 0.95]
    let capacity = (num_nodes.ln_1p() + num_active.ln_1p()) / 10.0;
    capacity.clamp(0.5, 0.95)
}

/// Proxy memory estimate in MB based on genome structure.
pub fn estimate_memory_mb(candidate: &ArchitectureCandidate) -> f32 {
    // Each node contributes ~0.5 MB on average at the 128-dim baseline
    candidate.genome.nodes.len() as f32 * 0.5 + candidate.genome.connections.len() as f32 * 0.01
}

/// Proxy FLOPs estimate for a forward pass.
pub fn estimate_flops(candidate: &ArchitectureCandidate) -> u64 {
    // Very rough: each active connection involves ~1k multiply-add operations
    let active_conns = candidate
        .genome
        .connections
        .iter()
        .filter(|c| c.active)
        .count() as u64;
    active_conns * 1_000
}

// ── Early stopping ────────────────────────────────────────────────────────────

/// Decide whether evolution should stop early based on the recent fitness trend.
///
/// Returns `true` if the best fitness has not improved by more than `min_delta` over the last
/// `patience` generations.
pub fn should_stop_early(generation_best_fitness: &[f32], patience: usize, min_delta: f32) -> bool {
    if generation_best_fitness.len() < patience + 1 {
        return false;
    }
    let window = &generation_best_fitness[generation_best_fitness.len() - patience - 1..];
    let oldest = window[0];
    let newest = *window.last().expect("window must have elements");
    (newest - oldest).abs() < min_delta
}
