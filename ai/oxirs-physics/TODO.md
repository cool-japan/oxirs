# oxirs-physics TODO

Development roadmap for physics-informed digital twin simulation bridge.

## Current Status (0.1.0-rc.1)

**Lines of Code:** 906 total (725 code, 161 blank, 20 comments)

**Implemented:**
- [x] Simulation orchestration framework (`SimulationOrchestrator`)
- [x] SciRS2 thermal simulation (1D heat diffusion with RK4 solver)
- [x] Basic conservation law checking (energy, momentum, mass)
- [x] Provenance tracking (software version, parameters hash, execution time)
- [x] Error types (`PhysicsError` with comprehensive variants)
- [x] Basic parameter and result structures
- [x] Mock simulation fallback (when scirs2-integrate not available)

**Partially Implemented (Stubs):**
- [ ] Parameter extraction from RDF (only stub implementation)
- [ ] Result injection to RDF (only stub implementation)
- [ ] Dimensional analysis (skeleton only)
- [ ] Digital twin management (basic structure)

## Phase 1: Core RDF Integration (High Priority)

### 1.1 Parameter Extraction from RDF
**File:** `src/simulation/parameter_extraction.rs`

- [ ] Implement SPARQL queries to extract entity properties
  - [ ] Query initial conditions (temperature, velocity, pressure)
  - [ ] Query boundary conditions (forces, heat flux, constraints)
  - [ ] Query material properties (conductivity, density, elasticity)
  - [ ] Query geometry/dimensions (length, area, volume)
- [ ] Parse RDF literals to Rust types with unit conversion
  - [ ] Use `scirs2_core::units::convert` for unit conversions
  - [ ] Handle XSD datatypes (decimal, double, integer)
  - [ ] Parse custom unit literals (e.g., "300 K", "1.5 MPa")
- [ ] Validate extracted parameters
  - [ ] Check for missing required properties
  - [ ] Validate physical constraints (positive mass, valid temperature range)
  - [ ] Verify unit consistency using `UnitChecker`
- [ ] Error handling for malformed RDF
  - [ ] Return `PhysicsError::ParameterExtraction` with context
  - [ ] Suggest fixes for common errors (missing units, wrong types)

**Estimated Complexity:** Medium (200-300 lines)

### 1.2 Result Injection to RDF
**File:** `src/simulation/result_injection.rs`

- [ ] Generate SPARQL UPDATE queries for simulation results
  - [ ] Insert state trajectory (time series data)
  - [ ] Insert derived quantities (stress, strain, flow rate)
  - [ ] Insert convergence metadata (iterations, residual)
  - [ ] Insert provenance (software version, timestamp, parameters hash)
- [ ] Link results to entity IRI with unique simulation run ID
  - [ ] Use UUID for simulation run IRI
  - [ ] Link via W3C PROV ontology (`prov:wasGeneratedBy`)
- [ ] Batch insert optimization for large trajectories
  - [ ] Use SPARQL INSERT DATA for batch operations
  - [ ] Chunk large trajectories (e.g., 1000 states per query)
- [ ] Transaction support (rollback on validation failure)
  - [ ] Use SPARQL transactions if supported by backend
  - [ ] Store temporary results before final commit

**Estimated Complexity:** Medium (250-350 lines)

### 1.3 Integration Tests
**File:** `tests/integration/rdf_roundtrip.rs`

- [ ] Create test RDF dataset with battery thermal properties
- [ ] Test extract → simulate → inject roundtrip
- [ ] Verify injected results match simulation output
- [ ] Test provenance tracking (parameters hash, timestamp)
- [ ] Test error handling (missing properties, invalid units)

**Estimated Complexity:** Small (100-150 lines)

## Phase 2: SAMM Aspect Model Support (High Priority)

### 2.1 SAMM Parser
**File:** `src/samm/parser.rs`

- [ ] Parse SAMM Aspect Model TTL files
  - [ ] Extract entity definitions (aspects, properties, characteristics)
  - [ ] Parse structured data types (collections, entities)
  - [ ] Handle SAMM units and constraints
- [ ] Convert SAMM properties to `SimulationParameters`
  - [ ] Map SAMM characteristics to physical quantities
  - [ ] Convert SAMM units to `uom` types
  - [ ] Extract constraints (min/max, pattern, range)
- [ ] Validate against SAMM meta-model
  - [ ] Check required properties
  - [ ] Verify datatype consistency

**Estimated Complexity:** Large (400-500 lines)

### 2.2 SAMM-to-RDF Bridge
**File:** `src/samm/rdf_bridge.rs`

- [ ] Convert SAMM aspect instances to RDF triples
- [ ] Support SAMM namespace resolution
- [ ] Handle SAMM inheritance and composition
- [ ] Generate SAMM-compliant RDF output

**Estimated Complexity:** Medium (200-300 lines)

## Phase 3: Advanced Simulations (Medium Priority)

### 3.1 Mechanical Simulation (Structural Analysis)
**File:** `src/simulation/scirs2_mechanical.rs`

- [ ] Finite Element Method (FEM) implementation
  - [ ] Use `scirs2_linalg` for stiffness matrix assembly
  - [ ] Solve linear systems (static analysis)
  - [ ] Compute stress, strain, displacement
- [ ] Material models
  - [ ] Linear elastic (Hooke's law)
  - [ ] Hyperelastic (rubber, biological tissue)
  - [ ] Plastic (yield criteria)
- [ ] Boundary conditions
  - [ ] Fixed displacement (Dirichlet)
  - [ ] Applied force (Neumann)
  - [ ] Contact (penalty method)
- [ ] Validation
  - [ ] Check stress-strain consistency
  - [ ] Verify momentum conservation
  - [ ] Compare with analytical solutions (beam bending, plate deflection)

**Estimated Complexity:** Very Large (600-800 lines)

### 3.2 Fluid Dynamics Simulation (CFD)
**File:** `src/simulation/scirs2_fluid.rs`

- [ ] Navier-Stokes solver
  - [ ] Use `scirs2_neural` for turbulence modeling (PINN)
  - [ ] Implement pressure-velocity coupling (SIMPLE algorithm)
  - [ ] Support incompressible and compressible flows
- [ ] Turbulence models
  - [ ] k-epsilon (RANS)
  - [ ] Large Eddy Simulation (LES) with `scirs2_neural`
- [ ] Boundary conditions
  - [ ] Inlet/outlet (velocity, pressure)
  - [ ] Wall (no-slip, slip)
  - [ ] Symmetry
- [ ] Validation
  - [ ] Check mass conservation (continuity equation)
  - [ ] Verify momentum conservation
  - [ ] Compare with benchmark cases (lid-driven cavity, channel flow)

**Estimated Complexity:** Very Large (700-900 lines)

### 3.3 Electrical Simulation (Circuit Analysis)
**File:** `src/simulation/scirs2_electrical.rs`

- [ ] Modified Nodal Analysis (MNA) for linear circuits
  - [ ] Use `scirs2_linalg` for sparse matrix solver
  - [ ] Support resistors, capacitors, inductors, voltage/current sources
- [ ] Nonlinear components (diodes, transistors)
  - [ ] Use `scirs2_optimize` for Newton-Raphson iteration
- [ ] Transient analysis (time-domain)
  - [ ] Use `scirs2_integrate` for ODE solving
- [ ] AC analysis (frequency-domain)
  - [ ] Complex number support via `scirs2_core::types::ComplexOps`
- [ ] Validation
  - [ ] Verify Kirchhoff's laws (current, voltage)
  - [ ] Check energy conservation
  - [ ] Compare with SPICE results

**Estimated Complexity:** Large (500-700 lines)

### 3.4 Multi-Physics Coupling
**File:** `src/simulation/coupled.rs`

- [ ] Thermo-mechanical coupling (thermal stress)
  - [ ] Iterate between thermal and mechanical solvers
  - [ ] Check convergence of coupled solution
- [ ] Electro-thermal coupling (Joule heating)
  - [ ] Electrical power → heat source
  - [ ] Temperature → resistivity update
- [ ] Fluid-structure interaction (FSI)
  - [ ] Pressure from fluid → force on structure
  - [ ] Deformation of structure → updated fluid domain
- [ ] Use `scirs2_core::parallel_ops` for concurrent solvers

**Estimated Complexity:** Very Large (800-1000 lines)

## Phase 4: Advanced Constraints (Medium Priority)

### 4.1 Full Dimensional Analysis
**File:** `src/constraints/dimensional_analysis.rs`

- [ ] Implement type-safe unit checking with `uom` crate
  - [ ] Define custom quantity types (ThermalConductivity, SpecificHeat)
  - [ ] Implement unit conversion between systems (SI, CGS, Imperial)
  - [ ] Validate equations have consistent dimensions
- [ ] Rayleigh method for dimensionless groups
  - [ ] Compute Pi groups (Reynolds, Nusselt, Prandtl numbers)
  - [ ] Use for simulation validation
- [ ] Buckingham Pi theorem implementation
  - [ ] Find dimensionless parameters from variable list
  - [ ] Suggest similarity laws

**Estimated Complexity:** Large (400-500 lines)

### 4.2 Advanced Conservation Laws
**File:** `src/constraints/conservation_laws.rs`

- [ ] Angular momentum conservation
  - [ ] Check torque balance in mechanical systems
- [ ] Entropy balance (2nd law of thermodynamics)
  - [ ] Verify entropy production ≥ 0
  - [ ] Use `scirs2_stats` for entropy calculation
- [ ] Charge conservation (electrical systems)
  - [ ] Verify Kirchhoff's current law at nodes
- [ ] Species conservation (chemical reactions)
  - [ ] Check mass balance for each species

**Estimated Complexity:** Medium (200-300 lines)

### 4.3 Physical Bounds Validation
**File:** `src/constraints/physical_bounds.rs` (new)

- [ ] Temperature bounds (0 K < T < melting point)
- [ ] Pressure bounds (P > 0 for real fluids)
- [ ] Stress bounds (von Mises < yield stress)
- [ ] Velocity bounds (v < speed of light, v < speed of sound for incompressible)
- [ ] Use `scirs2_core::validation::check_in_bounds`

**Estimated Complexity:** Small (150-200 lines)

## Phase 5: Digital Twin Synchronization (Medium Priority)

### 5.1 State Synchronization
**File:** `src/digital_twin/synchronization.rs` (new)

- [ ] Bidirectional sync between physical asset and digital twin
  - [ ] Pull sensor data from RDF → update simulation initial conditions
  - [ ] Push simulation predictions to RDF → update twin state
- [ ] Conflict resolution (sensor vs. simulation)
  - [ ] Use Kalman filter (implement via `scirs2_stats`)
  - [ ] Weighted average based on uncertainty
- [ ] Timestamp management
  - [ ] Track last sync time
  - [ ] Handle clock drift between physical and digital

**Estimated Complexity:** Medium (300-400 lines)

### 5.2 DTDL Support
**File:** `src/digital_twin/dtdl.rs` (new)

- [ ] Parse DTDL JSON files
  - [ ] Extract telemetry properties
  - [ ] Extract command interfaces
  - [ ] Extract relationship definitions
- [ ] Map DTDL to OxiRS RDF ontology
  - [ ] Telemetry → RDF properties with time series
  - [ ] Commands → simulation triggers
  - [ ] Relationships → RDF links between twins
- [ ] Generate DTDL from RDF schema
  - [ ] Reverse mapping for DTDL export
- [ ] Azure Digital Twins API integration (optional)
  - [ ] Use `reqwest` for REST API calls
  - [ ] Authenticate with Azure AD

**Estimated Complexity:** Large (500-600 lines)

### 5.3 Predictive Maintenance
**File:** `src/digital_twin/predictive_maintenance.rs` (new)

- [ ] Remaining Useful Life (RUL) prediction
  - [ ] Run degradation simulations (fatigue, wear)
  - [ ] Estimate time to failure threshold
- [ ] Anomaly detection
  - [ ] Compare sensor data with simulation predictions
  - [ ] Flag deviations > threshold (use `scirs2_stats`)
- [ ] Maintenance schedule optimization
  - [ ] Use `scirs2_optimize` for cost minimization
  - [ ] Balance downtime vs. failure risk

**Estimated Complexity:** Large (400-500 lines)

## Phase 6: Performance Optimization (Low Priority, High Impact)

### 6.1 GPU Acceleration
**File:** `src/gpu/mod.rs` (new)

- [ ] GPU-accelerated linear algebra (FEM assembly)
  - [ ] Use `scirs2_core::gpu::{GpuContext, GpuBuffer}`
  - [ ] Implement sparse matrix-vector multiply on GPU
- [ ] GPU-accelerated ODE solver
  - [ ] Parallelize RK4 stages across GPU threads
  - [ ] Use `scirs2_core::tensor_cores` for mixed precision
- [ ] Benchmarks vs. CPU
  - [ ] Measure speedup for different problem sizes
  - [ ] Profile GPU memory usage

**Estimated Complexity:** Large (400-600 lines)

### 6.2 SIMD Optimization
**File:** Various simulation files

- [ ] Vectorize conservation law checks
  - [ ] Use `scirs2_core::simd::SimdArray`
  - [ ] Batch validation across multiple states
- [ ] Vectorize numerical integration
  - [ ] SIMD-accelerated RK4 step
  - [ ] Use `scirs2_core::simd_ops::auto_vectorize`

**Estimated Complexity:** Medium (scattered changes, 200-300 lines total)

### 6.3 Parallel Execution
**File:** `src/parallel/mod.rs` (new)

- [ ] Parallel parameter extraction (multiple entities)
  - [ ] Use `scirs2_core::parallel_ops::par_chunks`
  - [ ] Extract parameters for 1000+ entities in parallel
- [ ] Parallel simulation execution (ensemble runs)
  - [ ] Monte Carlo uncertainty quantification
  - [ ] Parameter sweep (sensitivity analysis)
- [ ] Load balancing
  - [ ] Use `scirs2_core::parallel::LoadBalancer`
  - [ ] Distribute work across CPU cores

**Estimated Complexity:** Medium (300-400 lines)

## Phase 7: AI-Augmented Physics (Low Priority)

### 7.1 Neural Network Corrections
**File:** `src/ai/neural_correction.rs` (new)

- [ ] Train neural network to correct simulation errors
  - [ ] Use `scirs2_neural` for model training
  - [ ] Input: simulation parameters + coarse simulation
  - [ ] Output: correction to fine simulation
- [ ] Physics-Informed Neural Networks (PINN)
  - [ ] Encode conservation laws as loss terms
  - [ ] Use `scirs2_neural` with custom loss functions
- [ ] Hybrid physics-ML models
  - [ ] Fast neural surrogate for expensive simulations
  - [ ] Fall back to physics solver when uncertainty high

**Estimated Complexity:** Very Large (700-900 lines)

### 7.2 Embedding-Based Similarity
**File:** `src/ai/embedding_similarity.rs` (new)

- [ ] Generate embeddings for simulation results
  - [ ] Use `oxirs-embed` to encode state trajectories
  - [ ] Use `scirs2_linalg` for cosine similarity
- [ ] Find similar past simulations
  - [ ] Vector search in RDF knowledge graph
  - [ ] Retrieve parameters/results for transfer learning
- [ ] Case-based reasoning
  - [ ] Adapt past solutions to new problems
  - [ ] Use `scirs2_optimize` for parameter adjustment

**Estimated Complexity:** Large (400-500 lines)

## Phase 8: Streaming and Real-Time (Low Priority)

### 8.1 Real-Time Simulation Updates
**File:** `src/streaming/realtime.rs` (new)

- [ ] Integrate with `oxirs-stream` for live data
  - [ ] Subscribe to sensor data streams (Kafka/NATS)
  - [ ] Update simulation initial conditions in real-time
- [ ] Incremental simulation
  - [ ] Resume from last state instead of full restart
  - [ ] Adaptive time stepping based on data rate
- [ ] Stream simulation results
  - [ ] Publish state updates to Kafka/NATS
  - [ ] Enable real-time dashboards

**Estimated Complexity:** Large (500-600 lines)

## Phase 9: Documentation and Examples (High Priority)

### 9.1 User Guide
**File:** `docs/user_guide.md` (new)

- [ ] Getting started tutorial
  - [ ] Install dependencies (SciRS2, oxirs-core)
  - [ ] Run first thermal simulation
- [ ] Simulation type guide
  - [ ] When to use thermal/mechanical/fluid/electrical
  - [ ] Parameter requirements for each type
- [ ] RDF integration guide
  - [ ] How to structure RDF for parameter extraction
  - [ ] Example SPARQL queries
- [ ] SAMM integration guide
  - [ ] How to define SAMM Aspect Models
  - [ ] Mapping SAMM to simulation parameters

**Estimated Complexity:** N/A (documentation)

### 9.2 API Reference
**File:** Auto-generated via `cargo doc`

- [ ] Add comprehensive doc comments to all public items
- [ ] Include examples in doc comments
- [ ] Add module-level documentation with architecture diagrams

**Estimated Complexity:** Medium (scattered across codebase)

### 9.3 Example Programs
**Directory:** `examples/`

- [ ] `examples/basic_thermal.rs` - Basic thermal simulation
- [ ] `examples/battery_digital_twin.rs` - Battery thermal model with RDF
- [ ] `examples/mechanical_stress.rs` - Structural analysis (FEM)
- [ ] `examples/fluid_flow.rs` - CFD simulation
- [ ] `examples/multi_physics.rs` - Coupled thermo-mechanical
- [ ] `examples/predictive_maintenance.rs` - RUL prediction
- [ ] `examples/samm_integration.rs` - SAMM Aspect Model parsing

**Estimated Complexity:** Small (100-200 lines each, 700-1400 total)

## Phase 10: Testing and Quality (Continuous)

### 10.1 Unit Tests
- [ ] Increase test coverage to >80%
  - [ ] Test all error paths
  - [ ] Test boundary cases (zero, infinity, negative)
  - [ ] Test unit conversions
- [ ] Property-based testing with `proptest`
  - [ ] Conservation laws always hold
  - [ ] Dimensional analysis always consistent

**Estimated Complexity:** Continuous

### 10.2 Integration Tests
**Directory:** `tests/`

- [ ] RDF roundtrip tests (extract → simulate → inject)
- [ ] SAMM parsing tests (TTL → SimulationParameters)
- [ ] Multi-simulation orchestration tests
- [ ] Error recovery tests (malformed RDF, failed simulation)

**Estimated Complexity:** Medium (300-500 lines total)

### 10.3 Benchmarks
**Directory:** `benches/`

- [ ] Parameter extraction benchmark (1000 entities)
- [ ] Simulation performance benchmark (thermal, mechanical, fluid)
- [ ] Result injection benchmark (large trajectories)
- [ ] GPU vs CPU benchmark
- [ ] SIMD vs scalar benchmark

**Estimated Complexity:** Medium (200-300 lines total)

## Phase 11: Future Enhancements

### 11.1 Model Reduction
- [ ] Proper Orthogonal Decomposition (POD)
  - [ ] Use `scirs2_linalg` for SVD
  - [ ] Reduce large FEM models (1M DOF → 100 DOF)
- [ ] Reduced Basis Methods (RBM)
  - [ ] Greedy basis selection
  - [ ] Fast parameter queries

**Estimated Complexity:** Very Large (800-1000 lines)

### 11.2 Uncertainty Quantification
- [ ] Monte Carlo sampling
  - [ ] Use `scirs2_core::random` for sampling
  - [ ] Run 1000+ simulations with parameter variations
- [ ] Polynomial Chaos Expansion
  - [ ] Use `scirs2_stats` for orthogonal polynomials
  - [ ] Compute statistics (mean, variance, sensitivity)
- [ ] Bayesian inference
  - [ ] Update parameter distributions given observations
  - [ ] Use `scirs2_stats` for MCMC

**Estimated Complexity:** Very Large (700-900 lines)

### 11.3 Inverse Problems
- [ ] Parameter estimation from measurements
  - [ ] Use `scirs2_optimize` for minimization
  - [ ] Objective: minimize ||simulation - measurement||
- [ ] Topology optimization
  - [ ] Optimize material distribution (e.g., lightweight structure)
  - [ ] Use adjoint methods for gradient computation
- [ ] Shape optimization
  - [ ] Optimize geometry for desired performance

**Estimated Complexity:** Very Large (800-1000 lines)

## Summary

**Total Estimated Lines of Code (excluding docs):**
- Phase 1: ~800 lines
- Phase 2: ~700 lines
- Phase 3: ~3000 lines
- Phase 4: ~900 lines
- Phase 5: ~1300 lines
- Phase 6: ~1000 lines
- Phase 7: ~1400 lines
- Phase 8: ~600 lines
- Phase 9: ~1400 lines
- Phase 10: ~1000 lines
- Phase 11: ~2500 lines

**Total: ~13,600 lines** (current: 906 lines, ~15x growth)

**Recommended Implementation Order:**
1. Phase 1 (RDF integration) - Enables basic workflow
2. Phase 9 (documentation) - Makes usable for others
3. Phase 2 (SAMM) - Industry-standard ontology support
4. Phase 3 (simulations) - Core physics capabilities
5. Phase 4 (constraints) - Physics validation
6. Phase 5 (digital twins) - Real-world applications
7. Phase 6 (performance) - Production readiness
8. Phases 7-8, 11 (AI/streaming/advanced) - Future research

## Notes

- Follow [SCIRS2 Integration Policy](../../SCIRS2_INTEGRATION_POLICY.md)
- Use `scirs2-core` for all array/random operations (NO direct `ndarray`/`rand`)
- No warnings policy: `cargo clippy -- -D warnings`
- Refactor files >2000 lines using `splitrs`
- Use workspace dependencies (`*.workspace = true`)
- Update to latest crates.io versions
- All tests use `std::env::temp_dir()` for temporary files
