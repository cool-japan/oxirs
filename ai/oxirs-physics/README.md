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
│                    SciRS2 Simulation Engine                     │
│                                                                 │
│  Thermal:     Heat diffusion (scirs2-integrate ODE)             │
│  Mechanical:  Structural FEM (scirs2-linalg)                    │
│  Fluid:       Navier-Stokes CFD (scirs2-neural)                 │
│  Electrical:  Circuit analysis (scirs2-optimize)                │
│  Coupled:     Multi-physics (scirs2-parallel)                   │
│                                                                 │
│  Features:                                                      │
│  • GPU acceleration (scirs2-core::gpu)                          │
│  • SIMD vectorization (scirs2-core::simd)                       │
│  • Parallel execution (scirs2-core::parallel)                   │
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

- **Simulation Orchestration**: `SimulationOrchestrator` coordinates extract → run → inject workflow
- **Thermal Simulation**: 1D heat diffusion using SciRS2 ODE solvers (Runge-Kutta 4)
- **Conservation Laws**: Energy conservation validation for physics results
- **Provenance Tracking**: Full simulation metadata (software version, parameters hash, execution time)
- **Error Handling**: Comprehensive error types for physics operations

### Planned Features

See [TODO.md](TODO.md) for detailed roadmap.

- **Additional Simulation Types**: Mechanical, fluid dynamics, electrical, multi-physics
- **RDF Integration**: Full SPARQL parameter extraction and result injection
- **SAMM Support**: Parse SAMM Aspect Models for structured parameters
- **Advanced Constraints**: Full dimensional analysis with type-safe units
- **GPU Acceleration**: Large-scale simulations using scirs2-core::gpu
- **Streaming**: Real-time simulation updates via oxirs-stream
- **Hybrid Physics-ML**: Neural network corrections using oxirs-embed

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
oxirs-physics = { version = "0.1.0", features = ["simulation"] }
```

### Feature Flags

| Feature       | Description                              | Dependencies                    |
|--------------|------------------------------------------|---------------------------------|
| `simulation` | SciRS2-based physics simulations         | scirs2-integrate, scirs2-optimize, uom |
| `embeddings` | Neural network hybrid models             | scirs2-neural, oxirs-embed      |
| `samm`       | SAMM Aspect Model support                | oxirs-samm                      |
| `streaming`  | Real-time simulation updates             | oxirs-stream                    |
| `full`       | All features enabled                     | All of the above                |

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
use std::collections::HashMap;

fn validate_energy_conservation() -> Result<(), Box<dyn std::error::Error>> {
    let checker = ConservationChecker::new();

    let mut initial_state = HashMap::new();
    initial_state.insert("kinetic_energy".to_string(), 100.0);
    initial_state.insert("potential_energy".to_string(), 50.0);

    let mut final_state = HashMap::new();
    final_state.insert("kinetic_energy".to_string(), 75.0);
    final_state.insert("potential_energy".to_string(), 75.0);

    // Check energy conservation (should pass)
    checker.check(
        ConservationLaw::Energy,
        &initial_state,
        &final_state,
        1e-6  // tolerance
    )?;

    Ok(())
}
```

## Code Statistics

```bash
$ tokei .
===============================================================================
 Language            Files        Lines         Code     Comments       Blanks
===============================================================================
 Rust                   11          906          725           20          161
 |- Markdown            11          128           12          103           13
 (Total)                           1034          737          123          174
===============================================================================
```

## Project Structure

```
oxirs-physics/
├── src/
│   ├── lib.rs                           # Public API and module declarations
│   ├── error.rs                         # Error types (PhysicsError)
│   ├── simulation/
│   │   ├── mod.rs                       # SimulationOrchestrator
│   │   ├── parameter_extraction.rs      # RDF → SimulationParameters
│   │   ├── result_injection.rs          # SimulationResult → RDF
│   │   ├── simulation_runner.rs         # PhysicsSimulation trait
│   │   └── scirs2_thermal.rs            # Thermal simulation (SciRS2 ODE)
│   ├── constraints/
│   │   ├── mod.rs                       # Physics constraint API
│   │   ├── conservation_laws.rs         # Energy/momentum/mass conservation
│   │   └── dimensional_analysis.rs      # Unit checking (WIP)
│   └── digital_twin/
│       └── mod.rs                       # Digital twin management (WIP)
├── Cargo.toml                           # Dependencies and features
├── README.md                            # This file
└── TODO.md                              # Development roadmap
```

## SciRS2 Integration

`oxirs-physics` is built on the SciRS2 foundation and follows the [SciRS2 Integration Policy](../../SCIRS2_INTEGRATION_POLICY.md).

### Core Dependencies

| SciRS2 Crate         | Usage in oxirs-physics                          |
|---------------------|-------------------------------------------------|
| `scirs2-core`       | Array operations, random, SIMD, GPU, parallel   |
| `scirs2-integrate`  | ODE/PDE solvers for thermal/mechanical sims     |
| `scirs2-optimize`   | Parameter optimization, inverse problems        |
| `scirs2-neural`     | Neural network corrections for simulations      |
| `scirs2-linalg`     | Linear algebra for FEM/structural analysis      |
| `scirs2-stats`      | Statistical validation of simulation results    |

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

`oxirs-physics` will support Azure Digital Twins' DTDL for standardized twin definitions. See [TODO.md](TODO.md) for implementation roadmap.

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

Current version: `0.1.0`

Part of the OxiRS semantic web platform.
