//! Result Injection back to RDF

use crate::error::{PhysicsError, PhysicsResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Writes simulation results back to RDF graph
pub struct ResultInjector {
    /// Optional RDF store connection (for future SPARQL UPDATE)
    #[allow(dead_code)]
    store_path: Option<String>,

    /// Enable provenance tracking with W3C PROV ontology
    enable_provenance: bool,
}

impl ResultInjector {
    /// Create a new result injector
    pub fn new() -> Self {
        Self {
            store_path: None,
            enable_provenance: true,
        }
    }

    /// Create a result injector with RDF store connection
    pub fn with_store(store_path: impl Into<String>) -> Self {
        Self {
            store_path: Some(store_path.into()),
            enable_provenance: true,
        }
    }

    /// Disable provenance tracking
    pub fn without_provenance(mut self) -> Self {
        self.enable_provenance = false;
        self
    }

    /// Inject simulation results into RDF graph
    pub async fn inject(&self, result: &SimulationResult) -> PhysicsResult<()> {
        // For now, just validate the result structure
        // Future implementation will use SPARQL UPDATE to insert triples

        self.validate_result(result)?;

        // Mock injection - log what would be inserted
        tracing::debug!(
            "Would inject {} state vectors for entity {}",
            result.state_trajectory.len(),
            result.entity_iri
        );

        // Future: Execute SPARQL UPDATE
        // self.inject_with_sparql(result).await?;

        Ok(())
    }

    /// Validate result structure before injection
    fn validate_result(&self, result: &SimulationResult) -> PhysicsResult<()> {
        if result.entity_iri.is_empty() {
            return Err(PhysicsError::ResultInjection(
                "Entity IRI cannot be empty".to_string(),
            ));
        }

        if result.simulation_run_id.is_empty() {
            return Err(PhysicsError::ResultInjection(
                "Simulation run ID cannot be empty".to_string(),
            ));
        }

        if result.state_trajectory.is_empty() {
            return Err(PhysicsError::ResultInjection(
                "State trajectory cannot be empty".to_string(),
            ));
        }

        Ok(())
    }

    /// Inject results using SPARQL UPDATE (future implementation)
    #[allow(dead_code)]
    async fn inject_with_sparql(&self, result: &SimulationResult) -> PhysicsResult<()> {
        // Generate SPARQL UPDATE query for state trajectory
        let _update_query = self.generate_state_trajectory_update(result);

        // Generate provenance triples if enabled
        if self.enable_provenance {
            let _provenance_query = self.generate_provenance_update(result);
        }

        // TODO: Execute SPARQL UPDATE using oxirs-core::RdfStore
        // TODO: Handle batch inserts for large trajectories (chunk by 1000 states)
        // TODO: Use transactions for atomicity

        Err(PhysicsError::ResultInjection(
            "SPARQL UPDATE injection not yet implemented".to_string(),
        ))
    }

    /// Generate SPARQL UPDATE for state trajectory
    fn generate_state_trajectory_update(&self, result: &SimulationResult) -> String {
        let mut triples = Vec::new();

        // Create simulation run resource
        triples.push(format!(
            "<{}> a phys:SimulationRun .",
            result.simulation_run_id
        ));
        triples.push(format!(
            "<{}> phys:simulatesEntity <{}>",
            result.simulation_run_id, result.entity_iri
        ));
        triples.push(format!(
            "<{}> phys:timestamp \"{}\"^^xsd:dateTime .",
            result.simulation_run_id, result.timestamp
        ));

        // Add convergence info
        triples.push(format!(
            "<{}> phys:converged {} .",
            result.simulation_run_id, result.convergence_info.converged
        ));
        triples.push(format!(
            "<{}> phys:iterations {} .",
            result.simulation_run_id, result.convergence_info.iterations
        ));
        triples.push(format!(
            "<{}> phys:finalResidual {} .",
            result.simulation_run_id, result.convergence_info.final_residual
        ));

        // Add state trajectory (sample to avoid huge queries)
        for (idx, state) in result.state_trajectory.iter().enumerate().take(100) {
            let state_iri = format!("{}#state_{}", result.simulation_run_id, idx);
            triples.push(format!(
                "<{}> phys:hasState <{}> .",
                result.simulation_run_id, state_iri
            ));
            triples.push(format!("<{}> phys:time {} .", state_iri, state.time));

            for (key, value) in &state.state {
                triples.push(format!("<{}> phys:{} {} .", state_iri, key, value));
            }
        }

        format!(
            r#"
            PREFIX phys: <http://example.org/physics#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

            INSERT DATA {{
                {}
            }}
            "#,
            triples.join("\n                ")
        )
    }

    /// Generate SPARQL UPDATE for provenance (W3C PROV ontology)
    fn generate_provenance_update(&self, result: &SimulationResult) -> String {
        format!(
            r#"
            PREFIX prov: <http://www.w3.org/ns/prov#>
            PREFIX phys: <http://example.org/physics#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

            INSERT DATA {{
                <{}> prov:wasGeneratedBy <{}> .
                <{}> a prov:Activity .
                <{}> prov:startedAtTime "{}"^^xsd:dateTime .
                <{}> prov:used <{}> .
                <{}> prov:wasAssociatedWith <{}> .
                <{}> a prov:SoftwareAgent .
                <{}> prov:label "{}" .
                <{}> phys:version "{}" .
                <{}> phys:parametersHash "{}" .
            }}
            "#,
            result.entity_iri,
            result.simulation_run_id,
            result.simulation_run_id,
            result.simulation_run_id,
            result.provenance.executed_at,
            result.simulation_run_id,
            result.entity_iri,
            result.simulation_run_id,
            result.provenance.software,
            result.provenance.software,
            result.provenance.software,
            result.provenance.software,
            result.provenance.software,
            result.provenance.version,
            result.simulation_run_id,
            result.provenance.parameters_hash,
        )
    }

    /// Batch insert for large trajectories (future implementation)
    #[allow(dead_code)]
    async fn inject_in_batches(
        &self,
        _result: &SimulationResult,
        _batch_size: usize,
    ) -> PhysicsResult<()> {
        // TODO: Split state_trajectory into chunks of batch_size
        // TODO: Generate SPARQL UPDATE for each batch
        // TODO: Execute batches sequentially or in parallel
        Ok(())
    }
}

impl Default for ResultInjector {
    fn default() -> Self {
        Self::new()
    }
}

/// Simulation Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub entity_iri: String,
    pub simulation_run_id: String,
    pub timestamp: DateTime<Utc>,
    pub state_trajectory: Vec<StateVector>,
    pub derived_quantities: HashMap<String, f64>,
    pub convergence_info: ConvergenceInfo,
    pub provenance: SimulationProvenance,
}

/// State vector at a time point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateVector {
    pub time: f64,
    pub state: HashMap<String, f64>,
}

/// Convergence information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceInfo {
    pub converged: bool,
    pub iterations: usize,
    pub final_residual: f64,
}

/// Simulation provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationProvenance {
    pub software: String,
    pub version: String,
    pub parameters_hash: String,
    pub executed_at: DateTime<Utc>,
    pub execution_time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_result() -> SimulationResult {
        let mut trajectory = Vec::new();

        for i in 0..10 {
            let mut state = HashMap::new();
            state.insert("temperature".to_string(), 300.0 + i as f64);
            trajectory.push(StateVector {
                time: i as f64 * 10.0,
                state,
            });
        }

        SimulationResult {
            entity_iri: "urn:example:battery:001".to_string(),
            simulation_run_id: "run-123".to_string(),
            timestamp: Utc::now(),
            state_trajectory: trajectory,
            derived_quantities: HashMap::new(),
            convergence_info: ConvergenceInfo {
                converged: true,
                iterations: 100,
                final_residual: 1e-6,
            },
            provenance: SimulationProvenance {
                software: "oxirs-physics".to_string(),
                version: "0.1.0".to_string(),
                parameters_hash: "abc123".to_string(),
                executed_at: Utc::now(),
                execution_time_ms: 1500,
            },
        }
    }

    #[tokio::test]
    async fn test_result_injector_basic() {
        let injector = ResultInjector::new();
        let result = create_test_result();

        // Should succeed with valid result
        assert!(injector.inject(&result).await.is_ok());
    }

    #[tokio::test]
    async fn test_result_injector_with_store() {
        let injector = ResultInjector::with_store("./test_store");
        let result = create_test_result();

        assert!(injector.inject(&result).await.is_ok());
    }

    #[tokio::test]
    async fn test_result_injector_without_provenance() {
        let injector = ResultInjector::new().without_provenance();
        let result = create_test_result();

        assert!(injector.inject(&result).await.is_ok());
    }

    #[tokio::test]
    async fn test_validate_result_empty_entity_iri() {
        let injector = ResultInjector::new();
        let mut result = create_test_result();
        result.entity_iri = String::new();

        // Should fail with empty entity IRI
        assert!(injector.inject(&result).await.is_err());
    }

    #[tokio::test]
    async fn test_validate_result_empty_run_id() {
        let injector = ResultInjector::new();
        let mut result = create_test_result();
        result.simulation_run_id = String::new();

        // Should fail with empty run ID
        assert!(injector.inject(&result).await.is_err());
    }

    #[tokio::test]
    async fn test_validate_result_empty_trajectory() {
        let injector = ResultInjector::new();
        let mut result = create_test_result();
        result.state_trajectory.clear();

        // Should fail with empty trajectory
        assert!(injector.inject(&result).await.is_err());
    }

    #[test]
    fn test_generate_state_trajectory_update() {
        let injector = ResultInjector::new();
        let result = create_test_result();

        let query = injector.generate_state_trajectory_update(&result);

        // Check that query contains key elements
        assert!(query.contains("INSERT DATA"));
        assert!(query.contains("phys:SimulationRun"));
        assert!(query.contains(&result.simulation_run_id));
        assert!(query.contains(&result.entity_iri));
        assert!(query.contains("phys:converged"));
        assert!(query.contains("phys:iterations"));
    }

    #[test]
    fn test_generate_provenance_update() {
        let injector = ResultInjector::new();
        let result = create_test_result();

        let query = injector.generate_provenance_update(&result);

        // Check that query contains W3C PROV elements
        assert!(query.contains("prov:wasGeneratedBy"));
        assert!(query.contains("prov:Activity"));
        assert!(query.contains("prov:SoftwareAgent"));
        assert!(query.contains(&result.provenance.software));
        assert!(query.contains(&result.provenance.version));
        assert!(query.contains(&result.provenance.parameters_hash));
    }

    #[test]
    fn test_simulation_result_serialization() {
        let result = create_test_result();

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: SimulationResult = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.entity_iri, result.entity_iri);
        assert_eq!(deserialized.simulation_run_id, result.simulation_run_id);
        assert_eq!(
            deserialized.state_trajectory.len(),
            result.state_trajectory.len()
        );
        assert_eq!(
            deserialized.convergence_info.converged,
            result.convergence_info.converged
        );
    }

    #[test]
    fn test_state_vector_serialization() {
        let mut state = HashMap::new();
        state.insert("temperature".to_string(), 300.0);
        state.insert("pressure".to_string(), 101325.0);

        let vector = StateVector {
            time: 10.0,
            state: state.clone(),
        };

        let json = serde_json::to_string(&vector).unwrap();
        let deserialized: StateVector = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.time, 10.0);
        assert_eq!(deserialized.state.len(), 2);
        assert_eq!(deserialized.state.get("temperature"), Some(&300.0));
    }

    #[test]
    fn test_convergence_info() {
        let info = ConvergenceInfo {
            converged: true,
            iterations: 150,
            final_residual: 5e-7,
        };

        assert!(info.converged);
        assert_eq!(info.iterations, 150);
        assert!(info.final_residual < 1e-6);
    }

    #[test]
    fn test_provenance_tracking() {
        let prov = SimulationProvenance {
            software: "oxirs-physics".to_string(),
            version: "0.1.0-rc.1".to_string(),
            parameters_hash: "def456".to_string(),
            executed_at: Utc::now(),
            execution_time_ms: 2500,
        };

        assert_eq!(prov.software, "oxirs-physics");
        assert_eq!(prov.version, "0.1.0-rc.1");
        assert_eq!(prov.execution_time_ms, 2500);
    }
}
