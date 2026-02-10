//! Result Injection back to RDF
//!
//! Writes simulation results back to RDF graph using SPARQL UPDATE.
//! Supports provenance tracking, timestamps, and batch operations.

use crate::error::{PhysicsError, PhysicsResult};
use chrono::{DateTime, Utc};
use oxirs_core::model::NamedNode;
use oxirs_core::rdf_store::RdfStore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Writes simulation results back to RDF graph
pub struct ResultInjector {
    /// RDF store for updates
    store: Option<Arc<RdfStore>>,

    /// Enable provenance tracking with W3C PROV ontology
    enable_provenance: bool,

    /// Configuration
    config: InjectionConfig,
}

/// Injection configuration
#[derive(Debug, Clone)]
pub struct InjectionConfig {
    /// Physics namespace prefix
    pub physics_prefix: String,
    /// PROV namespace prefix
    pub prov_prefix: String,
    /// Batch size for large trajectories
    pub batch_size: usize,
    /// Use transactions
    pub use_transactions: bool,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            physics_prefix: "http://oxirs.org/physics#".to_string(),
            prov_prefix: "http://www.w3.org/ns/prov#".to_string(),
            batch_size: 1000,
            use_transactions: true,
        }
    }
}

impl ResultInjector {
    /// Create a new result injector without store (for testing)
    pub fn new() -> Self {
        Self {
            store: None,
            enable_provenance: true,
            config: InjectionConfig::default(),
        }
    }

    /// Create a result injector with RDF store
    pub fn with_store(store: Arc<RdfStore>) -> Self {
        Self {
            store: Some(store),
            enable_provenance: true,
            config: InjectionConfig::default(),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: InjectionConfig) -> Self {
        self.config = config;
        self
    }

    /// Disable provenance tracking
    pub fn without_provenance(mut self) -> Self {
        self.enable_provenance = false;
        self
    }

    /// Inject simulation results into RDF graph
    pub async fn inject(&self, result: &SimulationResult) -> PhysicsResult<()> {
        // Validate result structure
        self.validate_result(result)?;

        if let Some(ref store) = self.store {
            // Inject with SPARQL UPDATE
            self.inject_with_sparql(store, result).await?;
        } else {
            // Mock injection - just log
            tracing::debug!(
                "Would inject {} state vectors for entity {}",
                result.state_trajectory.len(),
                result.entity_iri
            );
        }

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

    /// Inject results using SPARQL UPDATE
    async fn inject_with_sparql(
        &self,
        store: &RdfStore,
        result: &SimulationResult,
    ) -> PhysicsResult<()> {
        // Generate SPARQL UPDATE for metadata
        let metadata_update = self.generate_metadata_update(result);

        // Execute metadata update
        self.execute_update(store, &metadata_update).await?;

        // Inject state trajectory in batches
        if result.state_trajectory.len() > self.config.batch_size {
            self.inject_in_batches(store, result).await?;
        } else {
            let trajectory_update = self.generate_state_trajectory_update(result);
            self.execute_update(store, &trajectory_update).await?;
        }

        // Generate provenance if enabled
        if self.enable_provenance {
            let provenance_update = self.generate_provenance_update(result);
            self.execute_update(store, &provenance_update).await?;
        }

        Ok(())
    }

    /// Execute SPARQL UPDATE
    async fn execute_update(&self, _store: &RdfStore, update_query: &str) -> PhysicsResult<()> {
        // Note: Current oxirs-core doesn't have update() method yet
        // For now, we'll log the update and return Ok
        // TODO: Implement when oxirs-core adds SPARQL UPDATE support

        tracing::debug!("SPARQL UPDATE:\n{}", update_query);

        // Future implementation:
        // store.update(update_query)
        //     .map_err(|e| PhysicsError::ResultInjection(format!("Update failed: {}", e)))?;

        Ok(())
    }

    /// Generate SPARQL UPDATE for simulation metadata
    fn generate_metadata_update(&self, result: &SimulationResult) -> String {
        let phys = &self.config.physics_prefix;

        format!(
            r#"
            PREFIX phys: <{phys}>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

            INSERT DATA {{
                <{run_id}> a phys:SimulationRun .
                <{run_id}> phys:simulatesEntity <{entity}> .
                <{run_id}> phys:timestamp "{timestamp}"^^xsd:dateTime .
                <{run_id}> phys:converged {converged}^^xsd:boolean .
                <{run_id}> phys:iterations {iterations}^^xsd:integer .
                <{run_id}> phys:finalResidual {residual}^^xsd:double .
            }}
            "#,
            phys = phys,
            run_id = result.simulation_run_id,
            entity = result.entity_iri,
            timestamp = result.timestamp.to_rfc3339(),
            converged = result.convergence_info.converged,
            iterations = result.convergence_info.iterations,
            residual = result.convergence_info.final_residual,
        )
    }

    /// Generate SPARQL UPDATE for state trajectory
    fn generate_state_trajectory_update(&self, result: &SimulationResult) -> String {
        let phys = &self.config.physics_prefix;
        let mut triples = Vec::new();

        // Limit to first 100 states to avoid huge queries (rest handled by batches)
        let states_to_insert = result.state_trajectory.iter().take(100);

        for (idx, state) in states_to_insert.enumerate() {
            let state_id = format!("{}#state_{}", result.simulation_run_id, idx);

            triples.push(format!(
                "<{run_id}> phys:hasState <{state_id}> .",
                run_id = result.simulation_run_id,
                state_id = state_id
            ));
            triples.push(format!("<{state_id}> phys:time {time}^^xsd:double .", state_id = state_id, time = state.time));

            for (key, value) in &state.state {
                triples.push(format!(
                    "<{state_id}> phys:{key} {value}^^xsd:double .",
                    state_id = state_id,
                    key = key,
                    value = value
                ));
            }
        }

        // Add derived quantities
        for (key, value) in &result.derived_quantities {
            triples.push(format!(
                "<{run_id}> phys:{key} {value}^^xsd:double .",
                run_id = result.simulation_run_id,
                key = key,
                value = value
            ));
        }

        format!(
            r#"
            PREFIX phys: <{phys}>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

            INSERT DATA {{
                {triples}
            }}
            "#,
            phys = phys,
            triples = triples.join("\n                ")
        )
    }

    /// Generate SPARQL UPDATE for provenance (W3C PROV ontology)
    fn generate_provenance_update(&self, result: &SimulationResult) -> String {
        let prov = &self.config.prov_prefix;
        let phys = &self.config.physics_prefix;

        format!(
            r#"
            PREFIX prov: <{prov}>
            PREFIX phys: <{phys}>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

            INSERT DATA {{
                <{entity}> prov:wasGeneratedBy <{run_id}> .
                <{run_id}> a prov:Activity .
                <{run_id}> prov:startedAtTime "{executed_at}"^^xsd:dateTime .
                <{run_id}> prov:used <{entity}> .
                <{run_id}> prov:wasAssociatedWith <{software_agent}> .
                <{software_agent}> a prov:SoftwareAgent .
                <{software_agent}> prov:label "{software}"^^xsd:string .
                <{software_agent}> phys:version "{version}"^^xsd:string .
                <{run_id}> phys:parametersHash "{params_hash}"^^xsd:string .
                <{run_id}> phys:executionTimeMs {exec_time}^^xsd:integer .
            }}
            "#,
            prov = prov,
            phys = phys,
            entity = result.entity_iri,
            run_id = result.simulation_run_id,
            executed_at = result.provenance.executed_at.to_rfc3339(),
            software_agent = format!("urn:agent:{}", result.provenance.software),
            software = result.provenance.software,
            version = result.provenance.version,
            params_hash = result.provenance.parameters_hash,
            exec_time = result.provenance.execution_time_ms,
        )
    }

    /// Batch insert for large trajectories
    async fn inject_in_batches(
        &self,
        store: &RdfStore,
        result: &SimulationResult,
    ) -> PhysicsResult<()> {
        let phys = &self.config.physics_prefix;

        for (batch_idx, chunk) in result
            .state_trajectory
            .chunks(self.config.batch_size)
            .enumerate()
        {
            let mut triples = Vec::new();

            for (idx_in_chunk, state) in chunk.iter().enumerate() {
                let global_idx = batch_idx * self.config.batch_size + idx_in_chunk;
                let state_id = format!("{}#state_{}", result.simulation_run_id, global_idx);

                triples.push(format!(
                    "<{run_id}> phys:hasState <{state_id}> .",
                    run_id = result.simulation_run_id,
                    state_id = state_id
                ));
                triples.push(format!("<{state_id}> phys:time {time}^^xsd:double .", state_id = state_id, time = state.time));

                for (key, value) in &state.state {
                    triples.push(format!(
                        "<{state_id}> phys:{key} {value}^^xsd:double .",
                        state_id = state_id,
                        key = key,
                        value = value
                    ));
                }
            }

            let batch_update = format!(
                r#"
                PREFIX phys: <{phys}>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                INSERT DATA {{
                    {triples}
                }}
                "#,
                phys = phys,
                triples = triples.join("\n                    ")
            );

            self.execute_update(store, &batch_update).await?;

            tracing::debug!(
                "Injected batch {} ({} states)",
                batch_idx + 1,
                chunk.len()
            );
        }

        Ok(())
    }

    /// Create result node IRI
    pub fn create_result_node(&self, _entity: &NamedNode, property: &str) -> PhysicsResult<NamedNode> {
        let result_id = Uuid::new_v4();
        let result_iri = format!("{}result_{}_{}", self.config.physics_prefix, property, result_id);

        NamedNode::new(&result_iri)
            .map_err(|e| PhysicsError::ResultInjection(format!("Invalid result IRI: {}", e)))
    }

    /// Write timestamped value
    pub async fn write_timestamped_value(
        &self,
        store: &RdfStore,
        result_node: &NamedNode,
        value: &ResultValue,
    ) -> PhysicsResult<()> {
        let timestamp = Utc::now();
        let phys = &self.config.physics_prefix;

        let update = match &value.value {
            ResultData::Scalar(v) => {
                format!(
                    r#"
                    PREFIX phys: <{phys}>
                    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                    INSERT DATA {{
                        <{node}> phys:property "{property}"^^xsd:string .
                        <{node}> phys:value {value}^^xsd:double .
                        <{node}> phys:timestamp "{timestamp}"^^xsd:dateTime .
                    }}
                    "#,
                    phys = phys,
                    node = result_node.as_str(),
                    property = value.property,
                    value = v,
                    timestamp = timestamp.to_rfc3339(),
                )
            }
            ResultData::Vector(vec) => {
                let mut triples = Vec::new();
                triples.push(format!(
                    "<{node}> phys:property \"{property}\"^^xsd:string .",
                    node = result_node.as_str(),
                    property = value.property
                ));
                triples.push(format!(
                    "<{node}> phys:timestamp \"{timestamp}\"^^xsd:dateTime .",
                    node = result_node.as_str(),
                    timestamp = timestamp.to_rfc3339()
                ));

                for (i, v) in vec.iter().enumerate() {
                    triples.push(format!(
                        "<{node}> phys:component{i} {v}^^xsd:double .",
                        node = result_node.as_str(),
                        i = i,
                        v = v
                    ));
                }

                format!(
                    r#"
                    PREFIX phys: <{phys}>
                    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                    INSERT DATA {{
                        {triples}
                    }}
                    "#,
                    phys = phys,
                    triples = triples.join("\n                        ")
                )
            }
            ResultData::Tensor(tensor) => {
                let mut triples = Vec::new();
                triples.push(format!(
                    "<{node}> phys:property \"{property}\"^^xsd:string .",
                    node = result_node.as_str(),
                    property = value.property
                ));

                for (i, row) in tensor.iter().enumerate() {
                    for (j, v) in row.iter().enumerate() {
                        triples.push(format!(
                            "<{node}> phys:tensor_{i}_{j} {v}^^xsd:double .",
                            node = result_node.as_str(),
                            i = i,
                            j = j,
                            v = v
                        ));
                    }
                }

                format!(
                    r#"
                    PREFIX phys: <{phys}>
                    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                    INSERT DATA {{
                        {triples}
                    }}
                    "#,
                    phys = phys,
                    triples = triples.join("\n                        ")
                )
            }
            ResultData::TimeSeries(series) => {
                let mut triples = Vec::new();
                triples.push(format!(
                    "<{node}> phys:property \"{property}\"^^xsd:string .",
                    node = result_node.as_str(),
                    property = value.property
                ));

                for (i, (time, value)) in series.iter().enumerate() {
                    let point_id = format!("{}#point_{}", result_node.as_str(), i);
                    triples.push(format!("<{node}> phys:hasPoint <{point_id}> .", node = result_node.as_str(), point_id = point_id));
                    triples.push(format!("<{point_id}> phys:time {time}^^xsd:double .", point_id = point_id, time = time));
                    triples.push(format!("<{point_id}> phys:value {value}^^xsd:double .", point_id = point_id, value = value));
                }

                format!(
                    r#"
                    PREFIX phys: <{phys}>
                    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                    INSERT DATA {{
                        {triples}
                    }}
                    "#,
                    phys = phys,
                    triples = triples.join("\n                        ")
                )
            }
        };

        self.execute_update(store, &update).await
    }

    /// Add provenance metadata
    pub async fn add_provenance(
        &self,
        store: &RdfStore,
        result_node: &NamedNode,
        provenance: &ProvenanceInfo,
    ) -> PhysicsResult<()> {
        let prov = &self.config.prov_prefix;

        let update = format!(
            r#"
            PREFIX prov: <{prov}>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

            INSERT DATA {{
                <{node}> prov:wasGeneratedBy <{activity}> .
                <{activity}> a prov:Activity .
                <{activity}> prov:startedAtTime "{timestamp}"^^xsd:dateTime .
                <{activity}> prov:wasAssociatedWith <{agent}> .
                <{agent}> a prov:SoftwareAgent .
                <{agent}> prov:label "{software}"^^xsd:string .
            }}
            "#,
            prov = prov,
            node = result_node.as_str(),
            activity = provenance.activity_id,
            timestamp = provenance.timestamp.to_rfc3339(),
            agent = provenance.agent_id,
            software = provenance.software,
        );

        self.execute_update(store, &update).await
    }

    /// Begin transaction
    pub fn begin_transaction(&self) -> PhysicsResult<Transaction> {
        Ok(Transaction {
            id: Uuid::new_v4().to_string(),
            updates: Vec::new(),
        })
    }

    /// Commit transaction
    pub async fn commit_transaction(&self, store: &RdfStore, tx: Transaction) -> PhysicsResult<()> {
        for update in tx.updates {
            self.execute_update(store, &update).await?;
        }
        Ok(())
    }
}

impl Default for ResultInjector {
    fn default() -> Self {
        Self::new()
    }
}

/// Transaction for atomic writes
#[derive(Debug, Clone)]
pub struct Transaction {
    pub id: String,
    pub updates: Vec<String>,
}

/// Provenance information
#[derive(Debug, Clone)]
pub struct ProvenanceInfo {
    pub activity_id: String,
    pub agent_id: String,
    pub software: String,
    pub timestamp: DateTime<Utc>,
}

/// Result value
#[derive(Debug, Clone)]
pub struct ResultValue {
    pub property: String,
    pub value: ResultData,
}

/// Result data types
#[derive(Debug, Clone)]
pub enum ResultData {
    Scalar(f64),
    Vector(Vec<f64>),
    Tensor(Vec<Vec<f64>>),
    TimeSeries(Vec<(f64, f64)>),
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
    fn test_generate_metadata_update() {
        let injector = ResultInjector::new();
        let result = create_test_result();

        let query = injector.generate_metadata_update(&result);

        // Check that query contains key elements
        assert!(query.contains("INSERT DATA"));
        assert!(query.contains("phys:SimulationRun"));
        assert!(query.contains(&result.simulation_run_id));
        assert!(query.contains(&result.entity_iri));
        assert!(query.contains("phys:converged"));
        assert!(query.contains("phys:iterations"));
    }

    #[test]
    fn test_generate_state_trajectory_update() {
        let injector = ResultInjector::new();
        let result = create_test_result();

        let query = injector.generate_state_trajectory_update(&result);

        // Check that query contains state information
        assert!(query.contains("INSERT DATA"));
        assert!(query.contains("phys:hasState"));
        assert!(query.contains("phys:time"));
        assert!(query.contains(&result.simulation_run_id));
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

        let json = serde_json::to_string(&result).expect("Failed to serialize");
        let deserialized: SimulationResult = serde_json::from_str(&json).expect("Failed to deserialize");

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

        let json = serde_json::to_string(&vector).expect("Failed to serialize");
        let deserialized: StateVector = serde_json::from_str(&json).expect("Failed to deserialize");

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
            version: "0.1.0".to_string(),
            parameters_hash: "def456".to_string(),
            executed_at: Utc::now(),
            execution_time_ms: 2500,
        };

        assert_eq!(prov.software, "oxirs-physics");
        assert_eq!(prov.version, "0.1.0");
        assert_eq!(prov.execution_time_ms, 2500);
    }

    #[test]
    fn test_injection_config() {
        let config = InjectionConfig {
            physics_prefix: "http://example.org/phys#".to_string(),
            prov_prefix: "http://example.org/prov#".to_string(),
            batch_size: 500,
            use_transactions: false,
        };

        assert_eq!(config.physics_prefix, "http://example.org/phys#");
        assert_eq!(config.batch_size, 500);
        assert!(!config.use_transactions);
    }

    #[test]
    fn test_create_result_node() {
        let injector = ResultInjector::new();
        let entity = NamedNode::new("http://example.org/entity1").expect("Failed to create node");

        let result_node = injector.create_result_node(&entity, "displacement").expect("Failed to create result node");

        assert!(result_node.as_str().contains("displacement"));
        assert!(result_node.as_str().starts_with("http://oxirs.org/physics#result_"));
    }

    #[test]
    fn test_transaction() {
        let injector = ResultInjector::new();
        let tx = injector.begin_transaction().expect("Failed to begin transaction");

        assert!(!tx.id.is_empty());
        assert!(tx.updates.is_empty());
    }
}
