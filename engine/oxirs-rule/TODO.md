# OxiRS Rule - TODO

*Last Updated: October 4, 2025*

## ✅ Current Status: v0.1.0-alpha.2 Released (Experimental)

**oxirs-rule** provides rule-based reasoning engine for RDF data (experimental feature).

### Alpha.2 Release Status (October 4, 2025)
- **89 tests passing** (unit + integration) with zero compilation warnings
- **RETE network** tuned with SciRS2 metrics & tracing
- **Forward/backward chaining** over persisted datasets and federated query outputs
- **RDFS/OWL RL reasoning** expanded with new inference profiles
- **CLI + server integration** for automatic inference refresh (cron + on-demand)
- **Released on crates.io**: `oxirs-rule = "0.1.0-alpha.2"` (experimental)

## 🎯 Post-Alpha Development Roadmap

### Beta Release Targets (v0.1.0-beta.1 - December 2025)

#### Reasoning Engine
- [ ] Complete RDFS reasoning
- [ ] OWL 2 RL profile
- [ ] Custom rule language
- [ ] Rule optimization

#### Performance
- [ ] Incremental reasoning
- [ ] Parallel rule execution
- [ ] Memory usage optimization
- [ ] Materialization strategies

#### Features
- [ ] Rule conflict resolution
- [ ] Explanation support
- [ ] Rule debugging tools
- [ ] Rule composition

#### Integration
- [ ] SPARQL integration
- [ ] Transaction support
- [ ] Distributed reasoning
- [ ] SHACL integration

### v0.2.0 Targets (Q1 2026)
- [ ] Advanced OWL reasoning
- [ ] Description Logic support
- [ ] Rule learning
- [ ] Probabilistic reasoning