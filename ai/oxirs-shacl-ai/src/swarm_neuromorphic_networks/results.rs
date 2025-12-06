//! Result types and metrics for swarm neuromorphic networks

use std::time::Duration;

#[derive(Debug)]
pub struct SwarmMetrics {
    pub total_swarm_validations: u64,
    pub total_processing_time: Duration,
    pub average_processing_efficiency: f64,
    pub uptime: Duration,
    pub communication_efficiency: f64,
}

impl SwarmMetrics {
    pub fn new() -> Self {
        Self {
            total_swarm_validations: 0,
            total_processing_time: Duration::from_secs(0),
            average_processing_efficiency: 0.0,
            uptime: Duration::from_secs(0),
            communication_efficiency: 0.0,
        }
    }
}

impl Default for SwarmMetrics {
    fn default() -> Self {
        Self::new()
    }
}
