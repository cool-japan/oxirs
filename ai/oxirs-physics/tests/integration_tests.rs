//! Integration Tests for OxiRS Physics
//!
//! Tests the full workflow: parameter extraction → simulation → result injection

use oxirs_physics::{
    constraints::{BoundsValidator, ConservationChecker, UnitChecker},
    simulation::{
        ParameterExtractor, PhysicsSimulation, ResultInjector, SciRS2ThermalSimulation,
        SimulationOrchestrator,
    },
};
use std::sync::Arc;

#[tokio::test]
async fn test_full_thermal_simulation_workflow() {
    // Step 1: Extract parameters
    let extractor = ParameterExtractor::new();
    let params = extractor
        .extract("urn:example:battery:thermal-test", "thermal")
        .await
        .expect("Parameter extraction should succeed");

    assert_eq!(params.simulation_type, "thermal");
    assert!(!params.initial_conditions.is_empty());

    // Step 2: Validate parameters
    let validator = BoundsValidator::new();
    validator
        .validate_parameters(&params)
        .expect("Parameters should be valid");

    // Step 3: Run simulation
    let simulation = SciRS2ThermalSimulation::default();
    let result = simulation
        .run(&params)
        .await
        .expect("Simulation should complete");

    assert!(!result.state_trajectory.is_empty());
    assert!(result.convergence_info.converged);

    // Step 4: Validate physics constraints
    simulation
        .validate_results(&result)
        .expect("Results should satisfy physics constraints");

    // Step 5: Check conservation laws
    let checker = ConservationChecker::new(0.01);
    let violations = checker.check(&result.state_trajectory);
    assert!(
        violations.is_empty(),
        "No conservation law violations expected"
    );

    // Step 6: Inject results back to RDF
    let injector = ResultInjector::new();
    injector
        .inject(&result)
        .await
        .expect("Result injection should succeed");
}

#[tokio::test]
async fn test_orchestrator_workflow() {
    // Create orchestrator
    let mut orchestrator = SimulationOrchestrator::new();

    // Register thermal simulation
    let thermal_sim = Arc::new(SciRS2ThermalSimulation::default());
    orchestrator.register("thermal", thermal_sim);

    // Execute full workflow
    let result = orchestrator
        .execute_workflow("urn:example:battery:001", "thermal")
        .await
        .expect("Workflow should complete");

    assert_eq!(result.entity_iri, "urn:example:battery:001");
    assert!(!result.state_trajectory.is_empty());
    assert!(result.convergence_info.converged);
}

#[tokio::test]
async fn test_parameter_extraction_and_validation() {
    let extractor = ParameterExtractor::new();
    let validator = BoundsValidator::new();

    // Test thermal parameters
    let thermal_params = extractor
        .extract("urn:test:thermal", "thermal")
        .await
        .unwrap();

    assert!(validator.validate_parameters(&thermal_params).is_ok());

    // Test mechanical parameters
    let mechanical_params = extractor
        .extract("urn:test:mechanical", "mechanical")
        .await
        .unwrap();

    assert!(validator.validate_parameters(&mechanical_params).is_ok());
}

#[test]
fn test_unit_consistency_validation() {
    let checker = UnitChecker::new();

    // Thermal units
    assert!(checker.validate_unit("K").is_ok());
    assert!(checker.validate_unit("W/(m*K)").is_ok());
    assert!(checker.validate_unit("J/(kg*K)").is_ok());

    // Mechanical units
    assert!(checker.validate_unit("Pa").is_ok());
    assert!(checker.validate_unit("N").is_ok());
    assert!(checker.validate_unit("m").is_ok());

    // Unit compatibility
    assert!(checker.check_compatibility("K", "K").unwrap());
    assert!(checker.check_compatibility("Pa", "Pa").unwrap());

    // Incompatible units
    assert!(!checker.check_compatibility("K", "Pa").unwrap());
    assert!(!checker.check_compatibility("m", "kg").unwrap());
}

#[tokio::test]
async fn test_simulation_with_custom_parameters() {
    use oxirs_physics::simulation::parameter_extraction::{
        MaterialProperty, PhysicalQuantity, SimulationParameters,
    };
    use std::collections::HashMap;

    // Create custom parameters for aluminum
    let mut initial_conditions = HashMap::new();
    initial_conditions.insert(
        "temperature".to_string(),
        PhysicalQuantity {
            value: 350.0, // 350 K
            unit: "K".to_string(),
            uncertainty: Some(0.5),
        },
    );

    let mut material_properties = HashMap::new();
    material_properties.insert(
        "thermal_conductivity".to_string(),
        MaterialProperty {
            name: "Thermal Conductivity".to_string(),
            value: 237.0, // Aluminum
            unit: "W/(m*K)".to_string(),
        },
    );
    material_properties.insert(
        "specific_heat".to_string(),
        MaterialProperty {
            name: "Specific Heat".to_string(),
            value: 900.0, // Aluminum
            unit: "J/(kg*K)".to_string(),
        },
    );
    material_properties.insert(
        "density".to_string(),
        MaterialProperty {
            name: "Density".to_string(),
            value: 2700.0, // Aluminum
            unit: "kg/m^3".to_string(),
        },
    );

    let params = SimulationParameters {
        entity_iri: "urn:example:aluminum-block".to_string(),
        simulation_type: "thermal".to_string(),
        initial_conditions,
        boundary_conditions: Vec::new(),
        time_span: (0.0, 50.0),
        time_steps: 25,
        material_properties,
        constraints: Vec::new(),
    };

    // Validate parameters
    let validator = BoundsValidator::new();
    assert!(validator.validate_parameters(&params).is_ok());

    // Run simulation
    let simulation = SciRS2ThermalSimulation::new(237.0, 900.0, 2700.0);
    let result = simulation.run(&params).await.unwrap();

    // Validate results
    assert_eq!(result.state_trajectory.len(), 25);
    assert!(result.convergence_info.converged);
    assert!(simulation.validate_results(&result).is_ok());
}

#[tokio::test]
async fn test_result_injection_with_provenance() {
    use chrono::Utc;
    use oxirs_physics::simulation::result_injection::{
        ConvergenceInfo, SimulationProvenance, SimulationResult, StateVector,
    };
    use std::collections::HashMap;
    use uuid::Uuid;

    // Create a test result with full provenance
    let mut trajectory = Vec::new();
    for i in 0..20 {
        let mut state = HashMap::new();
        state.insert("temperature".to_string(), 300.0 + i as f64 * 0.5);
        trajectory.push(StateVector {
            time: i as f64 * 5.0,
            state,
        });
    }

    let result = SimulationResult {
        entity_iri: "urn:example:test-entity".to_string(),
        simulation_run_id: Uuid::new_v4().to_string(),
        timestamp: Utc::now(),
        state_trajectory: trajectory,
        derived_quantities: HashMap::new(),
        convergence_info: ConvergenceInfo {
            converged: true,
            iterations: 150,
            final_residual: 1e-7,
        },
        provenance: SimulationProvenance {
            software: "oxirs-physics".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            parameters_hash: "test_hash_abc123".to_string(),
            executed_at: Utc::now(),
            execution_time_ms: 3500,
        },
    };

    // Validate result structure
    let validator = BoundsValidator::new();
    assert!(validator.validate_results(&result).is_ok());

    // Inject with provenance
    let injector = ResultInjector::new();
    assert!(injector.inject(&result).await.is_ok());

    // Inject without provenance
    let injector_no_prov = ResultInjector::new().without_provenance();
    assert!(injector_no_prov.inject(&result).await.is_ok());
}

#[test]
fn test_conservation_law_validation() {
    use oxirs_physics::simulation::result_injection::StateVector;
    use std::collections::HashMap;

    let checker = ConservationChecker::new(0.01); // 1% tolerance

    // Create trajectory with conserved energy
    let mut trajectory = Vec::new();
    let total_energy = 1000.0;

    for i in 0..10 {
        let mut state = HashMap::new();
        // Energy is perfectly conserved
        state.insert("energy".to_string(), total_energy);
        state.insert("mass".to_string(), 50.0); // Mass conserved
        trajectory.push(StateVector {
            time: i as f64,
            state,
        });
    }

    let violations = checker.check(&trajectory);
    assert!(violations.is_empty(), "No violations expected");

    // Create trajectory with violated energy conservation
    let mut bad_trajectory = Vec::new();
    for i in 0..10 {
        let mut state = HashMap::new();
        // Energy increases linearly (violation)
        state.insert("energy".to_string(), total_energy + i as f64 * 100.0);
        bad_trajectory.push(StateVector {
            time: i as f64,
            state,
        });
    }

    let violations = checker.check(&bad_trajectory);
    assert!(!violations.is_empty(), "Should detect energy violation");
    assert_eq!(violations[0].law, "Energy Conservation");
}

#[tokio::test]
async fn test_error_handling() {
    use oxirs_physics::simulation::parameter_extraction::SimulationParameters;
    use std::collections::HashMap;

    let validator = BoundsValidator::new();

    // Test invalid time span (end before start)
    let mut params = SimulationParameters {
        entity_iri: "urn:test".to_string(),
        simulation_type: "thermal".to_string(),
        initial_conditions: HashMap::new(),
        boundary_conditions: Vec::new(),
        time_span: (100.0, 0.0), // Invalid
        time_steps: 10,
        material_properties: HashMap::new(),
        constraints: Vec::new(),
    };

    assert!(validator.validate_parameters(&params).is_err());

    // Test zero time steps
    params.time_span = (0.0, 100.0);
    params.time_steps = 0; // Invalid
    assert!(validator.validate_parameters(&params).is_err());

    // Test valid parameters
    params.time_steps = 10;
    assert!(validator.validate_parameters(&params).is_ok());
}
