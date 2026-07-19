# oxirs-physics

Physics-informed digital twin simulation bridge for OxiRS semantic web platform.

## Overview

`oxirs-physics` bridges the gap between semantic RDF knowledge graphs and SciRS2-powered physics simulations, enabling physics-informed AI reasoning and digital twin synchronization.

**Key Capabilities:**

- Extract simulation parameters from RDF graphs and SAMM Aspect Models
- Run physics simulations using SciRS2 (thermal, mechanical, fluid, electrical, etc.)
- Validate results against physics constraints (conservation laws, dimensional analysis)
- Inject simulation results back to RDF with full provenance tracking
- Synchronize physical asset state with digital twin representations

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                      RDF Knowledge Graph                        │
│  • Entity properties (mass, dimensions, material)               │
│  • Initial conditions (temperature, pressure, velocity)         │
│  • Boundary conditions (constraints, forces, heat flux)         │
│  • SAMM Aspect Models (structured domain ontologies)            │
└───────────────┬─────────────────────────────────────────────────┘
                │ extract_parameters
                ▼
┌─────────────────────────────────────────────────────────────────┐
│               Parameter Extractor (SPARQL queries)              │
│  • Parse RDF properties → SimulationParameters                  │
│  • SAMM model interpretation → structured data                  │
│  • Unit conversion & validation                                 │
└───────────────┬─────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Physics Simulation Engine                    │
│                                                                 │
│  Thermal:     Heat diffusion (scirs2-integrate ODE, RK4)        │
│  Mechanical:  Structural FEM, modal & vibration analysis        │
│  Fluid:       Navier-Stokes CFD, pipe flow, Bernoulli, drag     │
│  Electrical:  Modified Nodal Analysis circuit simulation        │
│  Coupled:     Statistical mech., thermo., celestial, quantum    │
│                                                                 │
│  Features (in-house on scirs2-core, opt-in):                    │
│  • GPU acceleration (scirs2-core::gpu, feature `gpu`)            │
│  • PINN residual correction (feature `pinn_correction`)         │
└───────────────┬─────────────────────────────────────────────────┘
                │ SimulationResult
                ▼
┌─────────────────────────────────────────────────────────────────┐
│               Physics Constraint Validation                     │
│  • Conservation laws (energy, momentum, mass)                   │
│  • Dimensional analysis (unit consistency)                      │
│  • Physical bounds (temperature, pressure limits)               │
│  • Numerical stability (convergence checks)                     │
└───────────────┬─────────────────────────────────────────────────┘
                │ validated results
                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Result Injector (SPARQL UPDATE)                    │
│  • Insert state trajectory to RDF                               │
│  • Add derived quantities (stress, strain, flow rate)           │
│  • Record provenance (software version, parameters hash)        │
│  • Link to simulation run metadata (timestamp, convergence)     │
└───────────────┬─────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Updated RDF Knowledge Graph                        │
│  • Simulation results (time series data)                        │
│  • Provenance trail (reproducibility)                           │
│  • Digital twin state synchronized                              │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### Core Features (Implemented)

- **Simulation Orchestration**: `SimulationOrchestrator` coordinates extract → run → inject workflow across registered simulation backends
- **Multi-Domain Simulations**: Thermal (1D heat diffusion, RK4), structural (FEM stress analysis, modal analysis, vibration analysis), fluid dynamics (Navier-Stokes, pipe flow, Bernoulli, drag), electromagnetics (Modified Nodal Analysis), statistical mechanics, thermodynamics, celestial mechanics (N-body, Kepler, vis-viva), quantum mechanics (particle-in-box, QHO, tunneling, spin), optics (Snell, Fresnel, thin lens, diffraction), control systems (PID/cascade), kinematics, and 1D/2D wave propagation (FDTD)
- **RDF / SPARQL Integration**: SPARQL-based parameter extraction (`rdf::sparql_builder`, `simulation::parameter_extraction`) and SPARQL UPDATE result injection (`simulation::result_injection`), with RDF literal parsing and SI unit conversion (`rdf::literal_parser`)
- **SAMM Aspect Model Bridge**: Parses SAMM Aspect Model TTL and bridges it to simulation parameter types (`samm::physics_aspect`, `samm::fem_bridge`)
- **Physics Constraint Validation**: Conservation laws (energy, momentum, mass, with angular momentum/entropy checkers), Buckingham Pi dimensional analysis (`constraints::buckingham_pi`), and type-safe SI quantities via the `uom` crate (`simulation` feature)
- **Digital Twin & Bidirectional Sync**: RDF ↔ physics-state synchronization (`sync::rdf_to_state`, `sync::state_to_rdf`, `sync::bidirectional`) and a minimal versioned twin property store (`digital_twin::twin_value`)
- **DTDL v3 Support**: Parses Microsoft Digital Twin Definition Language v3 documents and maps them to/from RDF (`dtdl::{parser,mapper,validator,types}`)
- **Predictive Maintenance**: Remaining-useful-life prediction and anomaly detection (`predictive_maintenance`)
- **Material Property Database**: Reference material properties shared across simulation domains (`material_database`)
- **FEM & Adaptive Mesh Refinement**: Finite-element assembly and adaptive mesh refinement for structural/thermal analysis (`fem`, `mesh_refinement`)
- **Provenance Tracking**: Full simulation metadata (software version, parameters hash, execution time) attached to every result
- **Error Handling**: Comprehensive error types for physics operations
- **GPU-Accelerated Kernels** *(opt-in, `gpu` feature)*: FEM stress assembly, Navier-Stokes pressure solve, and heat-diffusion stencils dispatched to `scirs2_core::gpu`, with automatic CPU fallback (`GpuError::BackendUnavailable`) when no device is present
- **PINN Residual Correction** *(opt-in, `pinn_correction` feature)*: Pure-Rust feed-forward residual network applied online as a correction term to solver output (`pinn`) — no external ML framework dependency

### Planned / Future Work

See [TODO.md](TODO.md) for the detailed roadmap. Remaining open areas include real-time streaming integration (e.g. via `oxirs-stream`) and expanding PINN correction beyond the current single-residual-network scope.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
oxirs-physics = { version = "0.3.2", features = ["simulation"] }
```

### Feature Flags

| Feature          | Default | Description                                          | Dependencies              |
|------------------|:-------:|-------------------------------------------------------|----------------------------|
| `simulation`     |         | SciRS2 ODE solvers + type-safe `uom` quantities        | `scirs2-integrate`, `uom` |
| `gpu`             |         | GPU-accelerated FEM/Navier-Stokes/heat-diffusion kernels (CPU fallback always available) | `scirs2-core/gpu` |
| `pinn_correction` |         | Pure-Rust PINN residual correction network             | — (in-house)               |
| `full`            |         | Currently aliases `simulation`                          | `simulation`               |

Note: `samm`, `rdf`, `dtdl`, `sync`, and the various simulation-domain modules
(mechanical, fluid, electromagnetic, etc.) are unconditionally compiled — they
are not behind feature flags.

## Usage

### Basic Thermal Simulation

```rust
use oxirs_physics::simulation::{SimulationOrchestrator, SciRS2ThermalSimulation};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create orchestrator
    let mut orchestrator = SimulationOrchestrator::new();

    // Register thermal simulation
    let thermal_sim = Arc::new(SciRS2ThermalSimulation::default());
    orchestrator.register("thermal", thermal_sim);

    // Execute workflow: extract parameters from RDF → run simulation → inject results
    let result = orchestrator.execute_workflow(
        "urn:example:battery:001",  // Entity IRI
        "thermal"                    // Simulation type
    ).await?;

    println!("Simulation completed: {} states", result.state_trajectory.len());
    println!("Converged: {}", result.convergence_info.converged);

    Ok(())
}
```

### Custom Simulation with Full Control

```rust
use oxirs_physics::simulation::{
    ParameterExtractor, SimulationParameters, PhysicalQuantity,
    SciRS2ThermalSimulation, PhysicsSimulation, ResultInjector
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Setup parameters manually
    let mut initial_conditions = HashMap::new();
    initial_conditions.insert("temperature".to_string(), PhysicalQuantity {
        value: 300.0,
        unit: "K".to_string(),
        uncertainty: None,
    });

    let params = SimulationParameters {
        entity_iri: "urn:example:battery:thermal".to_string(),
        simulation_type: "thermal".to_string(),
        initial_conditions,
        boundary_conditions: Vec::new(),
        time_span: (0.0, 100.0),
        time_steps: 50,
        material_properties: HashMap::new(),
        constraints: Vec::new(),
    };

    // Step 2: Run simulation
    let sim = SciRS2ThermalSimulation::new(
        237.0,   // Thermal conductivity (W/m·K) - Aluminum
        900.0,   // Specific heat (J/kg·K) - Aluminum
        2700.0   // Density (kg/m³) - Aluminum
    );

    let result = sim.run(&params).await?;

    // Step 3: Validate physics constraints
    sim.validate_results(&result)?;

    // Step 4: Inject results to RDF
    let injector = ResultInjector::new();
    injector.inject(&result).await?;

    Ok(())
}
```

### Conservation Law Validation

```rust
use oxirs_physics::constraints::{ConservationChecker, ConservationLaw};
use oxirs_physics::simulation::result_injection::StateVector;
use std::collections::HashMap;

fn validate_energy_conservation() {
    // Checks Energy + Mass by default; add further laws with `add_law`.
    let mut checker = ConservationChecker::new(1e-6); // relative tolerance
    checker.add_law(ConservationLaw::Momentum);

    // A trajectory where total energy and mass stay constant across time steps.
    let trajectory: Vec<StateVector> = (0..10)
        .map(|i| {
            let mut state = HashMap::new();
            state.insert("energy".to_string(), 150.0); // kinetic + potential, constant
            state.insert("mass".to_string(), 50.0);
            StateVector { time: i as f64, state }
        })
        .collect();

    // No violations: energy and mass conservation hold within tolerance.
    let violations = checker.check(&trajectory);
    assert!(violations.is_empty());
}
```

## Code Statistics

```bash
$ tokei .
===============================================================================
 Language            Files        Lines         Code     Comments       Blanks
===============================================================================
 JSON                    3           98           98            0            0
 Markdown                2          348            0          272           76
 Rust                   78        36358        29562         2077         4719
 TOML                    1           57           34           12           11
-------------------------------------------------------------------------------
 Total                  84        36861        29694         2361         4806
===============================================================================
```

## Project Structure

```
oxirs-physics/
├── src/
│   ├── lib.rs                           # Public API and module declarations
│   ├── error.rs                         # Error types (PhysicsError)
│   ├── simulation/                      # SimulationOrchestrator, parameter extraction,
│   │   │                                # result injection, thermal (SciRS2 ODE / RK4)
│   │   └── {mod,parameter_extraction,result_injection,samm_parser,scirs2_thermal,simulation_runner}.rs
│   ├── constraints/                     # Conservation laws, Buckingham Pi, dimensional
│   │   └── {mod,buckingham_pi,conservation_laws,dimensional_analysis,physical_bounds}.rs
│   ├── conservation/                    # Extended conservation checkers (entropy, angular momentum)
│   │   └── {mod,checkers,checkers_impl,checkers_types,checkers_validator}.rs
│   ├── digital_twin/                    # Digital twin state + minimal versioned property store
│   │   └── {mod,twin_value}.rs
│   ├── dtdl/                            # DTDL v3 parser, RDF mapper, validator
│   │   └── {mod,parser,mapper,validator,types}.rs
│   ├── rdf/                             # SPARQL builder, RDF literal parsing/serialization
│   │   └── {mod,sparql_builder,literal_parser,physics_rdf*}.rs
│   ├── rdf_extraction/                  # RDF-graph parameter extraction helpers
│   ├── samm/                            # SAMM Aspect Model TTL → simulation bridge
│   │   └── {mod,physics_aspect,fem_bridge}.rs
│   ├── sync/                            # Bidirectional RDF ↔ physics-state synchronization
│   │   └── {mod,rdf_to_state,state_to_rdf,bidirectional}.rs
│   ├── gpu/                             # GPU kernels (feature `gpu`): FEM stress, Navier-Stokes, heat
│   │   └── {mod,stress_assembly,navier_stokes_kernel,heat_kernel}.rs
│   ├── pinn/                            # PINN residual correction (feature `pinn_correction`)
│   │   └── {mod,residual_model,loader,corrector}.rs
│   ├── fem/, mesh_refinement.rs         # Finite-element assembly + adaptive mesh refinement
│   ├── predictive_maintenance/          # RUL prediction, anomaly detection
│   ├── material_database.rs             # Reference material properties
│   ├── uom_quantities.rs                # Type-safe SI quantities via `uom` (feature `simulation`)
│   └── {celestial_mechanics,control_systems,electromagnetics,fluid_dynamics,
│         heat_transfer,kinematics,modal_analysis,optics,quantum_mechanics,
│         signal_processing,statistical_mechanics,stress_analysis,
│         thermal_analysis,thermal_system,thermodynamics,vibration_analysis,
│         wave_propagation}.rs           # Per-domain physics simulation modules
├── Cargo.toml                           # Dependencies and features
├── README.md                            # This file
└── TODO.md                              # Development roadmap
```

## SciRS2 Integration

`oxirs-physics` is built on the SciRS2 foundation and follows the [SciRS2 Integration Policy](../../SCIRS2_INTEGRATION_POLICY.md).

### Core Dependencies

| SciRS2 Crate         | Usage in oxirs-physics                                    |
|---------------------|--------------------------------------------------------------|
| `scirs2-core`       | Array operations, random, SIMD, GPU (feature `gpu`), parallel |
| `scirs2-integrate`  | ODE solvers for thermal/mechanical sims (feature `simulation`) |

FEM/structural analysis, statistical mechanics, and the PINN residual
corrector are implemented directly on `scirs2-core` (no `scirs2-optimize`,
`scirs2-neural`, `scirs2-linalg`, or `scirs2-stats` dependency).

### Full SciRS2 Usage Examples

```rust
// Arrays and numerical operations
use scirs2_core::ndarray_ext::{Array2, ArrayView2, array};
use scirs2_core::ndarray_ext::stats::mean;

// Random number generation
use scirs2_core::random::{Random, rng};

// Performance optimization
use scirs2_core::simd_ops::simd_dot_product;
use scirs2_core::parallel_ops::par_chunks;
use scirs2_core::gpu::{GpuContext, GpuBuffer};

// Memory efficiency for large RDF datasets
use scirs2_core::memory_efficient::MemoryMappedArray;
use scirs2_core::memory::BufferPool;

// Profiling and metrics
use scirs2_core::profiling::Profiler;
use scirs2_core::metrics::Timer;

// Error handling
use scirs2_core::error::{CoreError, Result};
```

## Development

### Build & Test

```bash
# Build with all features
cargo build --all-features

# Run tests (use nextest)
cargo nextest run --all-features

# Run with specific feature
cargo nextest run --features simulation

# Lint (no warnings policy)
cargo clippy --workspace --all-targets -- -D warnings

# Format check
cargo fmt --all -- --check
```

### Benchmarking

```bash
# Run benchmarks (when implemented)
cargo bench --features simulation
```

## Digital Twin Definition Language (DTDL)

`oxirs-physics` parses Azure Digital Twins' DTDL v3 documents and maps them to/from RDF (`dtdl::{parser, mapper, validator, types}`) for standardized twin definitions. See [TODO.md](TODO.md) for the roadmap.

Example DTDL integration:

```json
{
  "@context": "dtmi:dtdl:context;2",
  "@id": "dtmi:oxirs:Battery;1",
  "@type": "Interface",
  "contents": [
    {
      "@type": "Telemetry",
      "name": "temperature",
      "schema": "double",
      "unit": "kelvin"
    },
    {
      "@type": "Command",
      "name": "runThermalSimulation",
      "request": {
        "@type": "CommandPayload",
        "name": "parameters",
        "schema": "dtmi:oxirs:ThermalSimParams;1"
      }
    }
  ]
}
```

## Contributing

See the main [OxiRS README](../../README.md) for contribution guidelines.

**Development Guidelines:**

1. **SciRS2 First**: Use SciRS2 crates instead of direct `ndarray` or `rand` imports
2. **No Warnings**: All code must compile without warnings (`cargo clippy -- -D warnings`)
3. **Physics Validation**: All simulations must validate results against conservation laws
4. **Provenance**: All results must include full provenance metadata
5. **Testing**: Use `std::env::temp_dir()` for temporary files in tests
6. **Naming**: Use `snake_case` for variables, `PascalCase` for types

## References

### Physics Simulation

- **SciRS2**: `~/work/scirs/` - Scientific computing foundation
- **Apache Jena**: `~/work/jena/` - RDF/SPARQL reference
- **Oxigraph**: `~/work/oxigraph/` - RDF triple store reference

### Digital Twins

- **Azure DTDL**: [Digital Twins Definition Language](https://github.com/Azure/opendigitaltwins-dtdl)
- **ISO 23247**: [Digital Twin Framework for Manufacturing](https://www.iso.org/standard/75066.html)

### Standards

- **SAMM**: [Semantic Aspect Meta Model](https://eclipse-esmf.github.io/samm-specification/)
- **W3C PROV**: [Provenance Ontology](https://www.w3.org/TR/prov-o/)

## License

Same as OxiRS parent project (see repository root).

## Version

Current version: `0.4.0` (1,292 tests passing)

Part of the OxiRS semantic web platform.
