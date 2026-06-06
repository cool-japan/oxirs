# OxiRS Physics - TODO

*Version: 0.3.1 | Last Updated: June 6, 2026*

## Status: Production Ready

**oxirs-physics** provides physics-informed digital twin simulation bridge with SciRS2 integration.

### Features
- Simulation orchestration framework (SimulationOrchestrator)
- SciRS2 thermal simulation (1D heat diffusion with RK4 solver)
- Basic conservation law checking (energy, momentum, mass)
- Provenance tracking (software version, parameters hash, execution time)
- Comprehensive error types (PhysicsError with variants)
- Basic parameter and result structures
- Mock simulation fallback

### Current Limitations
- Parameter extraction from RDF (stub implementation)
- Result injection to RDF (stub implementation)
- Dimensional analysis (skeleton only)
- Digital twin management (basic structure)

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ SimulationOrchestrator, thermal simulation (RK4), conservation law checking, provenance tracking

### v0.2.3 - Current Release (March 16, 2026)
- ✅ SPARQL queries to extract entity properties
- ✅ RDF literals to Rust types with unit conversion
- ✅ SPARQL UPDATE queries for simulation results
- ✅ SAMM Aspect Model TTL parsing and SAMM-to-RDF bridge
- ✅ Mechanical simulation (stress analysis, FEM)
- ✅ Fluid dynamics simulation (Navier-Stokes)
- ✅ Electromagnetics simulation (Modified Nodal Analysis)
- ✅ Statistical mechanics, thermodynamics, celestial mechanics
- ✅ Kinematics, control systems, quantum mechanics
- ✅ Predictive maintenance (RUL prediction) and anomaly detection
- ✅ Digital twin framework, thermal analysis
- ✅ 1063 tests passing

### v0.4.0 - Planned (Q3 2026)
- [x] Full dimensional analysis with uom crate (planned 2026-04-17)
  - **Goal:** Integrate the uom crate for compile-time dimensional safety throughout physics calculations, replacing raw f64 parameters with typed dimensioned quantities
  - **Design:** Enable uom as non-optional dependency with SI system; add UomPhysicsQuantity wrapper; replace raw f64 in key physics modules with uom dimensioned types (Mass<SI>, Velocity<SI>, Energy<SI>, Temperature<SI>); update DimensionalAnalyzer to leverage uom's type system for unit consistency checking
  - **Files:** Cargo.toml, src/constraints/dimensional_analysis.rs, src/conservation/checkers.rs
  - **Prerequisites:** uom crate (latest from crates.io)
  - **Tests:** Unit mismatch compile-time error test; SI conversion correctness; dimensional consistency in conservation checkers
  - **Risk:** uom macro complexity; start with SI system only; may require updating function signatures across many modules
- [x] Advanced conservation laws (angular momentum, entropy) (planned 2026-04-17)
  - **Goal:** Add entropy (2nd law) and advanced angular momentum conservation checkers with Noether's theorem mapping
  - **Design:** EntropyConservationChecker validates dS >= 0 for closed systems and Clausius inequality (dQ/T <= dS); AngularMomentumChecker validates full 3D torque-angular momentum consistency (tau = dL/dt) and conservation in absence of external torque; NoetherSymmetryValidator maps symmetry type to conserved quantity (time translation -> energy, space translation -> momentum, rotation -> angular momentum)
  - **Files:** src/conservation/checkers.rs, src/conservation/entropy.rs (new), src/conservation/noether.rs (new)
  - **Tests:** Entropy monotonicity test on adiabatic process; angular momentum conservation on central force orbit; Noether mapping correctness for all three symmetries
  - **Risk:** Entropy calculation requires careful thermodynamic state tracking; validate against known analytical solutions
- [x] Buckingham Pi theorem implementation (implemented 2026-04-17)
  - **Goal:** Implement Buckingham Pi dimensional analysis — automatically identify dimensionless Pi groups from a set of physical variables using null-space computation over the dimensional matrix
  - **Design:** `BuckinghamPi::analyze(variables: &[PhysicalVar]) -> Vec<PiGroup>`; build dimensional matrix (rows=base units MLT, cols=variables); compute null space via Gaussian elimination with rational arithmetic to avoid floating-point pivoting errors; return Pi groups as exponent vectors; display as symbolic expressions; `PhysicalVar { name, dimensions: HashMap<BaseUnit, i32> }`
  - **Files:** src/constraints/buckingham_pi.rs (new), src/constraints/mod.rs
  - **Tests:** Classic pendulum (T, L, g, m) → 1 Pi group (T√(g/L)); Reynolds number recovery from fluid flow vars (ρ, v, L, μ); zero-dimensional variable passthrough; matrix rank deficiency detection
  - **Risk:** Null space over integers — use rational arithmetic (num-rational crate) to avoid floating-point pivoting errors
- [x] GPU acceleration for simulations (implemented 2026-04-30)
  - **Goal:** Apply the SAMM W3-S12 GPU pattern to physics: feature gate `gpu`, default off; CPU fallback via `scirs2_core`.
  - **Design:** Dispatch FEM kernels (stress assembly, mass matrix), Navier-Stokes pressure solve, and heat-diffusion stencils to GPU via `scirs2_core::gpu`. Pure-Rust default; opt-in C/Fortran via gpu feature.
  - **Files:** `src/gpu/{stress_assembly,navier_stokes_kernel,heat_kernel,mod}.rs`, `Cargo.toml` (gpu feature), `tests/gpu_kernels.rs`
  - **Prerequisites:** SAMM GPU pattern (round 1 W3-S12, already shipped)
  - **Tests:** unit GPU/CPU equality on small inputs; integration full FEM stress benchmark
  - **Risk:** GPU device availability in CI. Mitigation: skip integration when device absent, return GpuError::BackendUnavailable.
- [x] Neural network corrections (PINN) (implemented 2026-04-30)
  - **Goal:** Reduced-scope PINN — small residual neural network applied online as correction term to physics solver.
  - **Design:** Small ResNet-style 3-4 layers (~10K params) trained offline on physics-residual data; applied online as `state_t+1 = solver_step(state_t) + residual_nn(state_t)`. Tiny in-house feedforward — no PyTorch / scirs2-neural dep. Loaded from serialized model file via `serde_json` (NPZ writer not yet exposed by scirs2-core 0.4.2); feature `pinn_correction` (default off).
  - **Files:** `src/pinn/{residual_model,loader,corrector,mod}.rs`, `tests/pinn_correction.rs`
  - **Tests:** unit PINN correction on synthetic residual; integration heat-diffusion sim with vs without PINN
  - **Risk:** PINN training data not in scope (use synthetic for tests). Mitigation: ship example training script under `examples/`; runtime expects pre-trained weights.

### v1.0.0 - LTS Release (Q2 2026)
- [x] Bidirectional state synchronization (implemented 2026-04-30)
  - **Goal:** Add reverse direction to existing RDF → physics parameter extraction (now physics state → RDF triples).
  - **Design:** Existing physics ↔ RDF parameter extraction (one-way, RDF → physics) is in v0.2.3. Reverse direction now ships: physics state at simulation step t → RDF triples in a "state graph"; periodic sync via configurable interval; on RDF graph update, re-extract parameters and re-initialize. State diff implemented in-house (HashMap<String, f64> diff with tolerance) — no oxirs-aspect-differ dep needed for scalar diffs.
  - **Files:** `src/sync/{rdf_to_state,state_to_rdf,bidirectional,mod}.rs`, `tests/bidirectional_sync.rs`
  - **Tests:** unit on sync round-trip; integration full bidirectional sync over heat-diffusion sim
  - **Risk:** sync race conditions. Mitigation: lock state during sync window; use snapshot semantics.
- [x] DTDL (Digital Twin Definition Language) support (planned 2026-04-28)
  - **Goal:** Parse Microsoft DTDL v3 documents; map to RDF; round-trip via existing aspect-model machinery.
  - **Design:** Deserialize DTDL JSON-LD (Interface, Telemetry, Property, Command, Component, Relationship) → DtdlToRdfMapper → triples in oxirs-physics: namespace + QUDT units → RDF→DTDL reverse mapper. Validate DTMI IDs, version numbers, property types.
  - **Files:** src/dtdl/{parser,mapper,validator,types}.rs (new), src/lib.rs (export), tests/dtdl_round_trip.rs (new), tests/fixtures/dtdl/*.json (vendored samples)
  - **Tests:** unit per-element-type + DTMI validation + QUDT unit mapping + integration round-trip of sample DTDL docs
- [x] Long-term support guarantees (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)

## Notes

- Follow [SCIRS2 Integration Policy](../../SCIRS2_INTEGRATION_POLICY.md)
- Use `scirs2-core` for all array/random operations
- No warnings policy: `cargo clippy -- -D warnings`
- Use workspace dependencies (`*.workspace = true`)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS Physics v0.2.3 - Physics-informed digital twin simulation*
