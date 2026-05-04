# OxiRS Rule - TODO

*Version: 0.3.0 | Last Updated: May 3, 2026*

## Current Status

OxiRS Rule v0.2.3 is production-ready, providing rule-based reasoning with RDFS/OWL support and advanced rule formats.

### Production Features
- ✅ **RDFS Reasoning** - Complete RDFS entailment support
- ✅ **OWL Reasoning** - OWL 2 RL profile implementation
- ✅ **Custom Rules** - User-defined rule support
- ✅ **RIF Support** - Rule Interchange Format for cross-engine compatibility
- ✅ **CHR Support** - Constraint Handling Rules for declarative constraint solving
- ✅ **ASP Support** - Answer Set Programming for combinatorial optimization
- ✅ **2072 tests passing** with zero warnings

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ RDFS reasoning, OWL 2 RL, custom rules, RIF/CHR/ASP support
- ✅ 2072 tests passing

### v0.2.3 - Current Release (March 16, 2026)
- ✅ Additional OWL 2 profiles
- ✅ Performance optimizations for large rulesets
- ✅ Enhanced reasoning strategies
- ✅ Rule debugging tools
- ✅ Probabilistic reasoning
- ✅ Temporal reasoning support
- ✅ Enhanced rule interchange
- ✅ Production monitoring

### v0.3.0 - Planned (Q2 2026)
- [x] Complete reasoning engine (completed 2026-04-30)
  - **Goal:** Audit existing reasoning against OWL 2 RL profile (already shipped) and add EL and QL profile coverage; close any RL gaps; add a combined RL+EL closure mode for hybrid TBoxes; add conformance suite runner.
  - **Design:** Add `OwlProfile` enum: `RL | EL | QL | RLEL | DL`. EL reasoner: standard CEL algorithm (completion-based) producing classified taxonomy. QL reasoner: rewrite queries to UCQs per W3C QL semantics. EL fragment: intersection, existential restrictions, role chains/hierarchies, transitive/reflexive roles. QL fragment: inverse properties, role hierarchies, intersection. Closures via fixed-point loop with semi-naive evaluation. Conformance suite runner against W3C OWL 2 test cases.
  - **Files:** `src/owl2/{el_reasoner,ql_reasoner,rlel_combined}.rs` (new), `src/owl2/profile.rs` (new — OwlProfile enum), `src/owl2/mod.rs` (extend dispatcher), `src/owl_dl/` (existing, unchanged), `tests/owl2_conformance.rs` (new)
  - **Tests:** unit on classic taxonomies (Pizza ontology subsets), EL CEL completeness against pellet reference, QL UCQ rewriting correctness; integration W3C OWL 2 conformance test suite — pass_rate >= 0.95 per profile (RL: 1.0, EL: ≥0.95, QL: ≥0.95)
  - **Risk:** EL/QL closure can blow up on complex ontologies. Mitigation: semi-naive evaluation + bounded queue + warning when fixed-point exceeds 30s.
- [~] Long-term support guarantees (policy: docs/policies/lts.md)
- [~] Enterprise features (policy: docs/policies/enterprise.md, decomposed items listed therein)
- [x] Comprehensive benchmarks (completed 2026-04-29)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS Rule v0.2.3 - Rule-based reasoning engine*
