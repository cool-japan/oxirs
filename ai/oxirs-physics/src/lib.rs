//! # OxiRS Physics - Physics-Informed Digital Twin Bridge
//!
//! [![Version](https://img.shields.io/badge/version-0.2.2-blue)](https://github.com/cool-japan/oxirs/releases)
//!
//! **Status**: Production Release (v0.2.2)
//!
//! Connects RDF knowledge graphs with SciRS2 physics simulations.
//!
//! # Features
//!
//! - **Parameter Extraction**: Extract simulation parameters from RDF graphs and SAMM Aspect Models
//! - **Result Injection**: Write simulation results back to RDF with provenance
//! - **Physics Constraints**: Validate results against conservation laws
//! - **Digital Twin Sync**: Synchronize physical asset state with digital representation
//! - **SPARQL Integration**: Build and execute SPARQL queries for entity properties
//! - **RDF Literal Parsing**: Parse typed literals with full SI unit conversion
//! - **SAMM Bridge**: Parse SAMM Aspect Model TTL and bridge to simulation types
//!
//! # Architecture
//!
//! ```text
//! [RDF Graph] ──extract──> [Simulation Params]
//!                               │
//!                               ▼
//!                      [SciRS2 Simulation]
//!                               │
//!                               ▼
//!                     [Simulation Results]
//!                               │
//!          ┌────────────────────┴────────────────┐
//!          ▼                                     ▼
//! [Physics Validation]              [Provenance Tracking]
//!          │                                     │
//!          ▼                                     ▼
//! [Result Injection] ─────────────> [RDF Graph Updated]
//! ```
//!
//! # Examples
//!
//! ```rust,no_run
//! use oxirs_physics::simulation::SimulationOrchestrator;
//! use oxirs_physics::digital_twin::DigitalTwin;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create orchestrator
//! let orchestrator = SimulationOrchestrator::new();
//!
//! // Extract parameters from RDF
//! let params = orchestrator.extract_parameters(
//!     "urn:example:battery:001",
//!     "thermal_simulation"
//! ).await?;
//!
//! // Run simulation
//! let result = orchestrator.run("thermal_simulation", params).await?;
//!
//! // Inject results back to RDF
//! orchestrator.inject_results(&result).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # SPARQL / RDF Integration
//!
//! ```rust
//! use oxirs_physics::rdf::sparql_builder::{PhysicsPropertyQuery, PhysicsProperty};
//! use oxirs_physics::rdf::literal_parser::{parse_rdf_literal, PhysicalUnit, convert_unit};
//!
//! // Build a SPARQL SELECT query
//! let query = PhysicsPropertyQuery::new("urn:example:motor:42")
//!     .with_property(PhysicsProperty::Mass)
//!     .with_property(PhysicsProperty::Temperature)
//!     .build_select_query();
//!
//! assert!(query.contains("SELECT"));
//!
//! // Parse a typed RDF literal
//! let value = parse_rdf_literal("9.80665 m/s^2", None).expect("parse failed");
//! assert_eq!(value.unit, PhysicalUnit::MetersPerSecondSquared);
//!
//! // Convert units
//! let in_g = convert_unit(&value, &PhysicalUnit::StandardGravity).expect("convert failed");
//! assert!((in_g.value - 1.0).abs() < 1e-5);
//! ```

pub mod conservation;
pub mod constraints;
pub mod digital_twin;
pub mod error;
pub mod fem;
pub mod modal_analysis;
pub mod predictive_maintenance;
pub mod rdf;
pub mod rdf_extraction;
pub mod samm;
pub mod simulation;

// v1.1.0 Material property database
pub mod material_database;

// v1.1.0: Thermal finite-element analysis (CST elements, Dirichlet/Neumann/Robin BCs)
pub mod thermal_analysis;

// v1.1.0 round 5: Adaptive mesh refinement for FEM
pub mod mesh_refinement;

// v1.1.0 round 6: Heat transfer simulation (Fourier / Newton / Stefan-Boltzmann)
pub mod heat_transfer;

// v1.1.0 round 7: Basic computational fluid dynamics (pipe flow, Bernoulli, drag)
pub mod fluid_dynamics;

// v1.1.0 round 11: Structural vibration and modal analysis (mass-spring-damper systems)
pub mod vibration_analysis;

// v1.1.0 round 12: Lumped-capacity thermal analysis (Fourier / Newton / Stefan-Boltzmann)
pub mod thermal_system;

// v1.1.0 round 13: 1D/2D wave propagation simulation (FDTD, standing/traveling/attenuated waves)
pub mod wave_propagation;

// v1.1.0 round 11: Basic quantum mechanics (particle-in-box, QHO, tunneling, spin, density matrices)
pub mod quantum_mechanics;

// v1.1.0 round 12: Optical physics (Snell, Fresnel, thin lens, diffraction, prism)
pub mod optics;

// v1.1.0 round 13: Statistical mechanics (Maxwell-Boltzmann, partition function, Fermi-Dirac, Bose-Einstein, mean free path)
pub mod statistical_mechanics;

// v1.1.0 round 14: Orbital mechanics and N-body simulation (Newtonian gravity, Kepler, vis-viva)
pub mod celestial_mechanics;

// v1.1.0 round 15: PID and cascade control system simulation
pub mod control_systems;

// v1.1.0 round 16: Kinematic equations for linear and rotational motion
pub mod kinematics;

pub use error::{PhysicsError, PhysicsResult};

/// Physics simulation bridge version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
