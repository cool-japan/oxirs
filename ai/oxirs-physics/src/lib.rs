//! # OxiRS Physics - Physics-Informed Digital Twin Bridge
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/cool-japan/oxirs/releases)
//!
//! **Status**: Production Release (v0.1.0)
//!
//! Connects RDF knowledge graphs with SciRS2 physics simulations.
//!
//! # Features
//!
//! - **Parameter Extraction**: Extract simulation parameters from RDF graphs and SAMM Aspect Models
//! - **Result Injection**: Write simulation results back to RDF with provenance
//! - **Physics Constraints**: Validate results against conservation laws
//! - **Digital Twin Sync**: Synchronize physical asset state with digital representation
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

pub mod constraints;
pub mod digital_twin;
pub mod error;
pub mod simulation;

pub use error::{PhysicsError, PhysicsResult};

/// Physics simulation bridge version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
