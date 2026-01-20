# OxiRS Physics - TODO

*Version: 0.1.0 | Last Updated: 2026-01-06*

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

## Future Roadmap

### v0.2.0 - Core RDF Integration (Q1 2026 - Expanded)
- [ ] SPARQL queries to extract entity properties
- [ ] Parse RDF literals to Rust types with unit conversion
- [ ] Generate SPARQL UPDATE queries for simulation results
- [ ] Transaction support for validation rollback
- [ ] Integration tests for RDF roundtrip
- [ ] SAMM Aspect Model TTL parsing
- [ ] SAMM-to-RDF bridge
- [ ] Mechanical simulation (FEM implementation)
- [ ] Fluid dynamics simulation (Navier-Stokes)
- [ ] Electrical simulation (Modified Nodal Analysis)
- [ ] Multi-physics coupling

### v0.4.0 - Advanced Constraints (Q3 2026)
- [ ] Full dimensional analysis with uom crate
- [ ] Advanced conservation laws (angular momentum, entropy)
- [ ] Physical bounds validation
- [ ] Buckingham Pi theorem implementation

### v1.0.0 - LTS Release (Q2 2026)
- [ ] Bidirectional state synchronization
- [ ] DTDL (Digital Twin Definition Language) support
- [ ] Predictive maintenance (RUL prediction)
- [ ] Anomaly detection
- [ ] GPU acceleration for simulations
- [ ] SIMD optimization
- [ ] Parallel execution
- [ ] Neural network corrections (PINN)
- [ ] Embedding-based similarity
- [ ] Real-time streaming integration

## Notes

- Follow [SCIRS2 Integration Policy](../../SCIRS2_INTEGRATION_POLICY.md)
- Use `scirs2-core` for all array/random operations
- No warnings policy: `cargo clippy -- -D warnings`
- Use workspace dependencies (`*.workspace = true`)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS Physics v0.1.0 - Physics-informed digital twin simulation*
