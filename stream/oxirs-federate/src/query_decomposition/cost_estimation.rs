//! Cost estimation for query execution plans
//!
//! This module provides sophisticated cost estimation algorithms for different
//! execution strategies, including network costs, join costs, and resource utilization.


use super::types::*;

impl CostEstimator {
    /// Estimate network transfer cost
    pub fn estimate_transfer_cost(&self, result_size: u64) -> f64 {
        // Cost based on data transfer volume
        let bytes_per_result = 100.0; // Average bytes per result
        let transfer_cost_per_kb = 0.1;

        let total_bytes = result_size as f64 * bytes_per_result;
        let total_kb = total_bytes / 1024.0;

        total_kb * transfer_cost_per_kb
    }

    /// Estimate cost for distributed execution
    pub fn estimate_distributed_cost(&self, steps: &[PlanStep], requires_join: bool) -> f64 {
        let mut total_cost = 0.0;

        // Sum individual step costs
        for step in steps {
            total_cost += step.estimated_cost;
            total_cost += self.estimate_transfer_cost(step.estimated_results);
        }

        // Add join cost if required
        if requires_join && steps.len() > 1 {
            let left_size = steps[0].estimated_results;
            let right_size = steps.get(1).map_or(1000, |s| s.estimated_results);
            total_cost += self.estimate_join_cost(left_size, right_size);
        }

        total_cost
    }

    /// Estimate optimization potential
    pub fn estimate_optimization_potential(
        &self,
        baseline_cost: f64,
        optimized_cost: f64,
    ) -> OptimizationPotential {
        let improvement_ratio = (baseline_cost - optimized_cost) / baseline_cost;

        let potential_level = if improvement_ratio > 0.5 {
            OptimizationLevel::High
        } else if improvement_ratio > 0.2 {
            OptimizationLevel::Medium
        } else if improvement_ratio > 0.0 {
            OptimizationLevel::Low
        } else {
            OptimizationLevel::None
        };

        OptimizationPotential {
            baseline_cost,
            optimized_cost,
            improvement_ratio,
            potential_level,
        }
    }

    /// Estimate resource utilization cost
    pub fn estimate_resource_cost(&self, cpu_usage: f64, memory_usage: u64) -> f64 {
        let cpu_cost = cpu_usage * 10.0; // Cost per CPU unit
        let memory_cost = (memory_usage as f64 / 1024.0 / 1024.0) * 2.0; // Cost per MB

        cpu_cost + memory_cost
    }

    /// Estimate latency impact
    pub fn estimate_latency_cost(&self, network_latency: f64, processing_time: f64) -> f64 {
        // Cost increases non-linearly with latency
        let latency_factor = (network_latency / 100.0).powi(2);
        let processing_factor = processing_time / 1000.0;

        // Use the public method instead of accessing private field
        let base_cost = 100.0; // Default base cost
        (latency_factor + processing_factor) * 10.0 // Default network cost factor
    }
}

/// Optimization potential analysis
#[derive(Debug, Clone)]
pub struct OptimizationPotential {
    pub baseline_cost: f64,
    pub optimized_cost: f64,
    pub improvement_ratio: f64,
    pub potential_level: OptimizationLevel,
}

/// Levels of optimization potential
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    None,
    Low,
    Medium,
    High,
}

/// Cost breakdown for detailed analysis
#[derive(Debug, Clone)]
pub struct CostBreakdown {
    pub base_execution_cost: f64,
    pub network_transfer_cost: f64,
    pub join_processing_cost: f64,
    pub resource_utilization_cost: f64,
    pub latency_penalty_cost: f64,
    pub total_cost: f64,
}

impl Default for CostBreakdown {
    fn default() -> Self {
        Self::new()
    }
}

impl CostBreakdown {
    pub fn new() -> Self {
        Self {
            base_execution_cost: 0.0,
            network_transfer_cost: 0.0,
            join_processing_cost: 0.0,
            resource_utilization_cost: 0.0,
            latency_penalty_cost: 0.0,
            total_cost: 0.0,
        }
    }

    pub fn calculate_total(&mut self) {
        self.total_cost = self.base_execution_cost
            + self.network_transfer_cost
            + self.join_processing_cost
            + self.resource_utilization_cost
            + self.latency_penalty_cost;
    }
}

/// Service performance profile for cost estimation
#[derive(Debug, Clone)]
pub struct ServicePerformanceProfile {
    pub service_id: String,
    pub average_response_time: f64,
    pub throughput_capacity: u64,
    pub reliability_score: f64,
    pub resource_efficiency: f64,
}

impl ServicePerformanceProfile {
    pub fn calculate_efficiency_factor(&self) -> f64 {
        (self.reliability_score * self.resource_efficiency) / self.average_response_time
    }
}
