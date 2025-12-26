//! Physics Simulation Orchestration

pub mod parameter_extraction;
pub mod result_injection;
pub mod scirs2_thermal;
pub mod simulation_runner;

pub use parameter_extraction::{ParameterExtractor, SimulationParameters};
pub use result_injection::{ResultInjector, SimulationResult};
pub use scirs2_thermal::SciRS2ThermalSimulation;
pub use simulation_runner::{PhysicsSimulation, SimulationRunner};

use crate::error::{PhysicsError, PhysicsResult};
use std::collections::HashMap;
use std::sync::Arc;

/// Simulation Orchestrator - coordinates parameter extraction, simulation, and result injection
pub struct SimulationOrchestrator {
    extractor: Arc<ParameterExtractor>,
    injector: Arc<ResultInjector>,
    simulations: HashMap<String, Arc<dyn PhysicsSimulation>>,
}

impl SimulationOrchestrator {
    /// Create a new orchestrator
    pub fn new() -> Self {
        Self {
            extractor: Arc::new(ParameterExtractor::new()),
            injector: Arc::new(ResultInjector::new()),
            simulations: HashMap::new(),
        }
    }

    /// Register a simulation type
    pub fn register(&mut self, name: impl Into<String>, simulation: Arc<dyn PhysicsSimulation>) {
        self.simulations.insert(name.into(), simulation);
    }

    /// Extract parameters from RDF graph
    pub async fn extract_parameters(
        &self,
        entity_iri: &str,
        simulation_type: &str,
    ) -> PhysicsResult<SimulationParameters> {
        self.extractor.extract(entity_iri, simulation_type).await
    }

    /// Run simulation
    pub async fn run(
        &self,
        simulation_type: &str,
        params: SimulationParameters,
    ) -> PhysicsResult<SimulationResult> {
        let simulation = self.simulations.get(simulation_type).ok_or_else(|| {
            PhysicsError::Simulation(format!("Unknown simulation type: {}", simulation_type))
        })?;

        simulation.run(&params).await
    }

    /// Inject results back to RDF
    pub async fn inject_results(&self, result: &SimulationResult) -> PhysicsResult<()> {
        self.injector.inject(result).await
    }

    /// Full simulation workflow: extract → run → inject
    pub async fn execute_workflow(
        &self,
        entity_iri: &str,
        simulation_type: &str,
    ) -> PhysicsResult<SimulationResult> {
        let params = self.extract_parameters(entity_iri, simulation_type).await?;
        let result = self.run(simulation_type, params).await?;
        self.inject_results(&result).await?;
        Ok(result)
    }
}

impl Default for SimulationOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let orchestrator = SimulationOrchestrator::new();
        assert!(orchestrator.simulations.is_empty());
    }
}
