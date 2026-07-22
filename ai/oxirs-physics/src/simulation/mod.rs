//! Physics Simulation Orchestration

pub mod parameter_extraction;
pub mod result_injection;
pub mod samm_parser;
pub mod scirs2_thermal;
pub mod simulation_runner;

pub use parameter_extraction::{ParameterExtractor, SimulationParameters};
pub use result_injection::{ResultInjector, SimulationResult};
pub use samm_parser::{AspectModel, SammParser};
pub use scirs2_thermal::SciRS2ThermalSimulation;
pub use simulation_runner::{PhysicsSimulation, SimulationRunner};

use crate::error::{PhysicsError, PhysicsResult};
use oxirs_core::rdf_store::RdfStore;
use std::collections::HashMap;
use std::sync::Arc;

/// Simulation Orchestrator - coordinates parameter extraction, simulation, and result injection
pub struct SimulationOrchestrator {
    extractor: Arc<ParameterExtractor>,
    injector: Arc<ResultInjector>,
    simulations: HashMap<String, Arc<dyn PhysicsSimulation>>,
    /// `true` when this orchestrator was built with a real backing RDF
    /// store ([`Self::with_store`]); `false` when it was built with
    /// [`Self::new`], in which case `extract_parameters` returns
    /// simulation-type-keyed mock parameters and `inject_results` is a
    /// logging no-op instead of touching any graph.
    has_store: bool,
}

impl SimulationOrchestrator {
    /// Create a new orchestrator with **no backing RDF store**.
    ///
    /// This is a mock/testing constructor: [`Self::extract_parameters`]
    /// falls back to hardcoded, simulation-type-keyed default parameters
    /// (ignoring `entity_iri` entirely) and [`Self::inject_results`] is a
    /// no-op that only logs a debug message rather than writing anything
    /// to RDF. Do not use this constructor in production code paths that
    /// need real parameter extraction or result persistence — use
    /// [`Self::with_store`] instead.
    pub fn new() -> Self {
        Self {
            extractor: Arc::new(ParameterExtractor::new()),
            injector: Arc::new(ResultInjector::new()),
            simulations: HashMap::new(),
            has_store: false,
        }
    }

    /// Create a new orchestrator backed by a real RDF store.
    ///
    /// This is the production constructor: [`Self::extract_parameters`]
    /// queries `store` via SPARQL for the entity's properties (falling
    /// back to defaults only for genuinely missing initial conditions,
    /// per `ExtractionConfig::use_defaults`), and
    /// [`Self::inject_results`] writes simulation results (state
    /// trajectory, convergence info, and W3C PROV provenance) back into
    /// `store` via SPARQL UPDATE.
    pub fn with_store(store: Arc<RdfStore>) -> Self {
        Self {
            extractor: Arc::new(ParameterExtractor::with_store(store.clone())),
            injector: Arc::new(ResultInjector::with_store(store)),
            simulations: HashMap::new(),
            has_store: true,
        }
    }

    /// `true` if this orchestrator is backed by a real RDF store (i.e. it
    /// was constructed with [`Self::with_store`]), `false` if it is
    /// operating in mock mode ([`Self::new`]).
    pub fn has_store(&self) -> bool {
        self.has_store
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
    use oxirs_core::query::{UpdateExecutor, UpdateParser};
    use oxirs_core::Store;

    #[test]
    fn test_orchestrator_creation() {
        let orchestrator = SimulationOrchestrator::new();
        assert!(orchestrator.simulations.is_empty());
        assert!(!orchestrator.has_store());
    }

    /// Regression test for the P0 finding: `SimulationOrchestrator` had no
    /// way to attach a real RDF store, so `extract_parameters` always fell
    /// back to fabricated mock parameters regardless of `entity_iri`, and
    /// `inject_results` always silently no-oped instead of writing to RDF.
    /// This proves `with_store` wires a real store through both extraction
    /// and injection.
    #[tokio::test]
    async fn regression_orchestrator_with_store_does_real_extraction() {
        let store = RdfStore::default();

        // Seed the store with a real triple for the entity under a
        // predicate name containing "initial" (not one of the mock's
        // hardcoded keys), so we can tell real extraction apart from the
        // canned mock parameters.
        let insert = r#"
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            INSERT DATA {
                <http://example.org/entity1> <http://oxirs.org/physics#initial_temperature> "310.5"^^xsd:double .
            }
        "#;
        let parser = UpdateParser::new();
        let parsed = parser.parse(insert).expect("parse failed");
        let executor = UpdateExecutor::new(&store);
        executor.execute(&parsed).expect("execute failed");

        let orchestrator = SimulationOrchestrator::with_store(Arc::new(store));
        assert!(orchestrator.has_store());

        let params = orchestrator
            .extract_parameters("http://example.org/entity1", "thermal")
            .await
            .expect("extraction failed");

        // Real extraction must surface the actual RDF-sourced key...
        let extracted = params
            .initial_conditions
            .get("initial_temperature")
            .expect("expected real RDF-extracted initial_temperature");
        assert!((extracted.value - 310.5).abs() < 1e-9);

        // ...and must NOT silently substitute the mock's canned
        // "temperature" = 293.15 key, since real (non-empty) initial
        // conditions were already found in the graph.
        assert!(
            !params.initial_conditions.contains_key("temperature"),
            "real extraction must not fall back to mock defaults when RDF data was found"
        );
    }

    /// Regression test proving `inject_results` on a store-backed
    /// orchestrator actually persists triples, rather than silently
    /// logging and no-oping as it always did before `with_store` existed.
    #[tokio::test]
    async fn regression_orchestrator_with_store_injects_real_triples() {
        let store = Arc::new(RdfStore::default());
        let orchestrator = SimulationOrchestrator::with_store(store.clone());

        let mut state = HashMap::new();
        state.insert("temperature".to_string(), 305.0);

        let result = SimulationResult {
            entity_iri: "http://example.org/entity1".to_string(),
            simulation_run_id: "urn:run:regression-1".to_string(),
            timestamp: chrono::Utc::now(),
            state_trajectory: vec![result_injection::StateVector { time: 0.0, state }],
            derived_quantities: HashMap::new(),
            convergence_info: result_injection::ConvergenceInfo {
                converged: true,
                iterations: 10,
                final_residual: 1e-8,
            },
            provenance: result_injection::SimulationProvenance {
                software: "oxirs-physics".to_string(),
                version: "0.4.1".to_string(),
                parameters_hash: "hash".to_string(),
                executed_at: chrono::Utc::now(),
                execution_time_ms: 5,
            },
        };

        assert!(store.find_quads(None, None, None, None).unwrap().is_empty());

        orchestrator
            .inject_results(&result)
            .await
            .expect("injection failed");

        let quads = store
            .find_quads(None, None, None, None)
            .expect("find_quads failed");
        assert!(
            !quads.is_empty(),
            "with_store orchestrator must actually write triples to the RDF store"
        );
    }
}
