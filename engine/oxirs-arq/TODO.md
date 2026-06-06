# OxiRS ARQ - TODO

*Version: 0.3.1 | Last Updated: June 6, 2026*

## Current Status

OxiRS ARQ v0.3.1 is production-ready, providing a SPARQL 1.1/1.2 query engine with histogram-based optimization and advanced caching.

### Production Features
- ✅ **SPARQL 1.1/1.2 Query Engine** - Full W3C compliance
- ✅ **Adaptive Query Optimization** - 3.8x faster via automatic complexity detection
- ✅ **Smart Query Batch Executor** - Parallel execution with priority queuing
- ✅ **Query Performance Analyzer** - ML-powered bottleneck detection
- ✅ **Cost-Based Optimizer** - Statistical cardinality estimation
- ✅ **Federation Support** - SERVICE clause execution
- ✅ **Query Caching** - Result caching with fingerprinting
- ✅ **SciRS2 Integration** - Full scientific computing compliance
- ✅ **2688 tests passing** with zero warnings

### Key Performance Metrics
- Query optimization: ~3.0 µs for all profiles (3.8x faster)
- 75% CPU savings at production scale (100K QPS)
- Zero overhead for complex queries

## Recent Accomplishments (v0.2.3)

### Performance Enhancements
- ✅ **Histogram-based Statistics** - Cost-based optimizer now uses histogram statistics for accurate cardinality estimation
- ✅ **TTL-based Cache Invalidation** - Smart caching with time-to-live and dependency tracking
- ✅ **Query Result Caching** - Advanced caching strategies with automatic invalidation

### Optimization Improvements
- ✅ **Statistical Cost Model** - Improved selectivity estimation using histograms
- ✅ **Cache Performance Metrics** - Comprehensive monitoring of cache hit rates and effectiveness
- ✅ **Adaptive Query Plans** - Runtime feedback for dynamic plan optimization

## Future Roadmap

### v0.3.0 - Advanced Query Execution (Q2 2026)
- [x] Adaptive join ordering with runtime feedback (planned 2026-04-17)
  - **Goal:** Wire AdaptiveJoinOrderOptimizer and AdaptiveStatsStore into the query executor so runtime cardinality feedback flows back to the planner and enables re-optimization
  - **Design:** Executor calls record_pattern_execution(pattern_id, estimated, actual) post-join; re-optimization hook triggers plan re-evaluation when correction factor diverges >2×; lock-free counter accumulation to minimize RwLock contention on RuntimeStats
  - **Files:** src/executor.rs, src/optimizer/adaptive.rs, src/adaptive_index_advisor.rs
  - **Tests:** Property-based tests on feedback convergence; benchmark showing plan improvement after 100 executions
  - **Risk:** Locking contention on Arc<RwLock<RuntimeStats>>; mitigate with lock-free counter accumulation
- [x] Query result materialized views (planned 2026-04-17)
  - **Goal:** Implement incremental/delta refresh for materialized views, multi-query view fusion, and change data capture integration
  - **Design:** DeltaRefresh strategy computes new_data - old_snapshot via set difference; ChangeDataCapture integration point for source tracking; multi-query view consolidation identifies shared subpatterns
  - **Files:** src/materialized_views.rs, src/optimizer/materialized_view.rs, src/optimizer/view_registry.rs
  - **Tests:** Incremental view staleness test; delta computation correctness vs full refresh; view fusion deduplication
  - **Risk:** Delta computation requires consistent snapshot isolation; use version timestamps
- [x] Parallel query execution across cores (planned 2026-04-17)
  - **Goal:** Add NUMA-aware thread scheduling to the parallel executor and dynamic work rebalancing during execution
  - **Design:** Parse /sys/devices/system/node/ for NUMA topology; pin Rayon threads per NUMA node via ParallelConfig { numa_awareness: bool }; dynamic work stealing with rebalancing when thread utilization diverges >20%
  - **Files:** src/parallel.rs
  - **Tests:** NUMA thread locality verification; parallel speedup benchmark; rebalancing under skewed workload
  - **Risk:** NUMA detection is Linux-specific; graceful fallback when unavailable
- [x] Distributed query planning (completed 2026-04-30)
  - **Goal:** Cost-based federation-aware planning that emits federated query plans across multiple SPARQL endpoints, leveraging the existing oxirs-federate orchestrator.
  - **Design:** When a query references an IRI within a known federated dataset, the planner emits `Algebra::Service { endpoint, pattern, silent }` nodes (re-using the existing SPARQL 1.1 SERVICE primitive instead of inventing a new variant). Cost model is supplied by an injectable `SourceSelectivityProvider` trait so `oxirs-arq` stays free of an `oxirs-federate` dependency (avoiding a future cycle); embedders bridge the trait to `oxirs-federate::source_selector::SourceSelector` at the integration layer. The bundled `StaticSourceProvider` covers the common in-process registration use case. Existing `arq::federation::FederationExecutor` decomposes Service nodes into subqueries for parallel dispatch (no new executor wire required).
  - **Files:** `src/optimizer/federated_plan.rs` (new — 824 lines), `src/lib.rs` (re-exports for `FederatedPlanner`, `SourceSelectivityProvider`, `StaticSourceProvider`, `FederatedSelectivity`).
  - **Prerequisites:** existing oxirs-federate API (already shipped); existing `arq::federation` dispatcher.
  - **Tests:** 9 unit tests in `optimizer::federated_plan::tests` (pattern → endpoint mapping, longest-prefix matching, silent semantics, cost-ordered Joins, recursion into Filter/LeftJoin, pass-through for pre-existing Service nodes); 8 integration tests in `tests/distributed_planning.rs` (3-endpoint cost ordering, mixed local/federated, custom provider impl).
  - **Risk:** federated subresult merging semantics. Mitigation: re-uses `Algebra::Service` so existing `FederationExecutor::merge_results` (SPARQL 1.1 Federation spec semantics) handles results.
  - **Refinement (2026-04-30):** Wired `FederatedPlanner` into the main `Optimizer::optimize()` flow as an opt-in pass. New API: `Optimizer::with_federated_planner(Arc<dyn SourceSelectivityProvider>)`, `Optimizer::with_federated_latency_weight(f64)`, `Optimizer::has_federated_planner() -> bool`, `Optimizer::last_federated_outcome() -> Option<&FederatedPlanOutcome>`. Pass runs after rule/cost-based passes so filter pushdown completes first; FILTER nodes end up sitting *outside* the emitted Service nodes, preserving SPARQL 1.1 SERVICE semantics. When no provider is registered, optimizer behavior is byte-for-byte identical to the pre-W2-S4 baseline.
  - **Bridge (2026-04-30):** Added `oxirs-federate::arq_bridge::ArqSourceSelectivityProvider`, a thin adapter wrapping `Arc<SourceSelector>` and implementing `SourceSelectivityProvider`. Endpoint selection consults VoID metadata (`property_partitions`, `class_partitions`, `uri_spaces`) and picks the highest-scored eligible source. Selectivity uses VoID partition counts (0.9 confidence), uri_space prefix matches (0.5 confidence), or default fallback (0.1 confidence). Auto-elevates to SILENT semantics when an endpoint reports `is_reachable == false`. Snapshot semantics — register sources before wrapping; mutations after wrapping are not observed (recommendation: build selector, then `Arc::new(selector)`). Bridge added `oxirs-arq.workspace = true` to `stream/oxirs-federate/Cargo.toml` (no cycle: `oxirs-federate` already depends on `oxirs-core` and `oxirs-vec`, neither of which depends on `oxirs-arq`).
  - **Refinement files:** `src/optimizer/mod.rs` (Optimizer struct + builder methods + integration in optimize()), `stream/oxirs-federate/src/arq_bridge.rs` (587 lines — new), `stream/oxirs-federate/Cargo.toml` (new oxirs-arq dep), `stream/oxirs-federate/src/lib.rs` (re-export), `stream/oxirs-federate/tests/arq_bridge_integration.rs` (494 lines — new). 7 new unit tests in `optimizer::federated_integration_tests` (no-provider passthrough, provider-emits-Service, mixed-local-and-federated, latency-weight-propagation, filter-after-federation, outcome-recorded). 14 new bridge unit tests in `arq_bridge::tests` (endpoint routing, selectivity computation, silent_default behavior, latency floor, exclusion, score-based tie-break, rdf:type via class_partitions). 8 new integration tests in `arq_bridge_integration` (3-endpoint federation, baseline regression, exclusion, score-based selection across two endpoints).
- [x] Advanced index selection (planned 2026-04-17)
  - **Goal:** Add runtime index creation/dropping, utilization tracking, and redundancy detection to the index advisor
  - **Design:** IndexStatisticsCollector tracks per-query index hits; runtime_index_create()/runtime_index_drop() triggered when recommendation threshold crossed; consolidation detects redundant SPO/SOP combos; multi-predicate index optimization
  - **Files:** src/adaptive_index_advisor.rs, src/query_analysis.rs, src/advanced_optimizer/index_advisor.rs
  - **Tests:** Index recommendation convergence test; redundancy detection on known duplicate indexes; multi-predicate selectivity test
  - **Risk:** Runtime DDL on indexes may block queries; use async index build
- [x] JIT compilation phase a — algebra-level plan cache (planned 2026-05-01)
  - **Goal:** First phase of JIT compilation. A query-plan cache keyed by algebra-fingerprint hash. Repeated identical queries skip the optimizer entirely. True JIT codegen (Cranelift/LLVM) is phases b–d, deferred. Phase a delivers measurable latency wins on hot queries on its own.
  - **Design:** New module `engine/oxirs-arq/src/plan_cache/{mod.rs,fingerprint.rs,cache.rs,eviction.rs}`. Algebra fingerprint: stable structural hash via post-order tree walk into `seahash::SeaHasher` (already workspace dep); variable names normalised (`?x → ?_v0`) so spelling-only differences collide; constants hashed by value. `PlanCache` — bounded LRU (default 1024 entries) with `parking_lot::RwLock` for concurrent reads. Cache hits skip optimizer; misses run optimization then insert. Schema-version-tick invalidation. `PlanCache::stats()` → `(hits, misses, evictions, avg_compile_us)`. Optimizer integration: hook into `Optimizer::optimize()` — fingerprint first, look up, return or run+insert.
  - **Files:** `engine/oxirs-arq/src/plan_cache/{mod.rs,fingerprint.rs,cache.rs,eviction.rs}`, `src/optimizer/mod.rs` (integration hook), `src/lib.rs` (re-export `pub mod plan_cache`), `examples/plan_cache_demo.rs`, `tests/plan_cache_test.rs`, `benches/plan_cache_bench.rs` (criterion).
  - **Prerequisites:** `parking_lot`, `seahash` — already in workspace.
  - **Tests:** renamed-variable queries collide → 1 cache entry; structurally distinct plans do not collide; capacity bound + LRU eviction; schema-tick clears cache; 8-thread concurrent-read stress test (no deadlock); round-trip returns same `CompiledPlan`; bench soft-asserts ≥10× speedup on hot query.
  - **Risk:** false collisions → wrong-plan execution. Mitigation: fingerprint covers structural shape + normalised variable bindings; correctness verified via round-trip equality tests for both collision and non-collision cases; cache returns clones, not internal references.

### v1.0.0 - LTS Release (Q2 2026)
- [x] Performance SLA guarantees (completed 2026-04-30)
  - **Goal:** Per-tenant SLA classes wired into the query executor with admission control + priority queueing, reusing the proven oxirs-vec W2-S6 SLA pattern, generalized into shared `oxirs-core::sla`.
  - **Design:** Moved SLA primitives from `oxirs-vec/src/multi_tenancy/{sla,admission_controller,priority_queue}.rs` into `oxirs-core/src/sla/`. New shared types: `SlaClass {Bronze, Silver, Gold, Platinum}`, `SlaThresholds`, `AdmissionController`, `AdmissionError`, `PriorityDispatcher` (renamed from `SlaQueryDispatcher`), `PrioritizedQuery`. The oxirs-vec multi_tenancy module re-exports the moved types via shim files; `SlaQueryDispatcher` is preserved as a back-compat type alias for `PriorityDispatcher`. The `From<AdmissionError> for MultiTenancyError` conversion stays inside `oxirs-vec` (orphan rule: target is local to vec). The ARQ executor exposes a new `execute_for_tenant(tenant, algebra, dataset)` method that calls `ArqSlaGate::admit(tenant, ())` → on accept enqueues into the priority dispatcher then dispatches via `execute()`; on reject returns typed `ArqSlaError::SlaExceeded` (with tier display name). Per-tenant config in `src/tenant_config.rs` (`TenantConfig` + `TenantConfigRegistry`).
  - **Files:** `core/oxirs-core/src/sla/{mod,class,thresholds,admission_controller,priority_dispatcher}.rs` (new), `core/oxirs-core/src/lib.rs` (`pub mod sla`), `engine/oxirs-arq/src/sla_integration.rs` (new — `ArqSlaGate`, `ArqSlaError`, `AdmittedQuery`, `DispatchedQuery`), `engine/oxirs-arq/src/tenant_config.rs` (new — `TenantConfig`, `TenantConfigRegistry`), `engine/oxirs-arq/src/executor/{queryexecutor_type,queryexecutor_new_group}.rs` (admission control wire via `with_sla_gate` + `execute_for_tenant`), `engine/oxirs-vec/src/multi_tenancy/{sla,admission_controller,priority_queue,mod}.rs` (re-export shims).
  - **Tests:** 18 unit tests in `oxirs_core::sla::*` (threshold ordering, dispatch priority, FIFO-within-tier, Bronze depletion, custom-cost admit), 18 unit tests in `oxirs_arq::sla_integration::*` + `tenant_config::*` (admit unregistered/registered, soft mode, dispatcher ordering, hot-add tenant), 9 integration tests in `tests/sla_admission.rs` (10-tenant × 4-class simulator, dispatcher class ordering, soft canary mode, registry clone semantics), 7 in oxirs-vec (back-compat `SlaQueryDispatcher` alias, `From<AdmissionError>` boundary impl).
  - **Risk:** Moving multi_tenancy types out of oxirs-vec without breaking downstream call sites. Mitigation: re-export from original paths — verified by full `oxirs-vec` test suite (1635 tests passing).
- [x] Complete SPARQL 1.2 compliance (planned 2026-04-17)
  - **Goal:** Audit against W3C SPARQL 1.2 working drafts, implement missing functions, add conformance test runner
  - **Design:** Systematic review of W3C SPARQL 1.2 draft spec; implement missing path syntax variants, graph management operations, and new built-in functions; automated conformance test suite runner against W3C test vectors
  - **Files:** src/query_validator.rs, src/algebra/, src/executor.rs, tests/sparql12_conformance.rs
  - **Tests:** W3C SPARQL 1.2 conformance test suite (all MUST assertions); regression tests for 1.1 compatibility
  - **Risk:** SPARQL 1.2 spec still evolving; pin to latest W3C draft date
- [x] Enterprise query optimization (planned 2026-05-01)
  - **Goal:** Runtime resource governor enforcing wall-time, result-row, and triple-scan budgets during query execution.
  - **Files:** `src/query_governor.rs`, `src/lib.rs` (re-export), `tests/resource_governor_test.rs`.
- [x] Advanced analytics integration (planned 2026-05-01)
  - **Goal:** Custom SPARQL-style aggregate functions backed by graph topology analytics (PageRank, betweenness, connected components, clustering coefficient, degree centrality).
  - **Files:** `src/analytics/{mod.rs,graph_analytics_agg.rs}`, `src/lib.rs` (re-export), `tests/analytics_test.rs`, `benches/analytics_bench.rs`.

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS ARQ v0.2.3 - High-performance SPARQL query engine with histogram optimization*

## Proposed follow-ups

- [x] JIT phase b — Cranelift-based filter expression codegen (planned 2026-05-01)
  - **Goal:** Compile SPARQL numeric filter expressions to native machine code via Cranelift, achieving 5-10× speedup on hot numeric-filter queries. Scope: filter expressions only (single operator family per plan).
  - **Supported subset:** numeric literals, variable loads, comparison operators (<, >, <=, >=, =, !=), arithmetic (+, -, *, /), logical (&&, ||, !), builtins (ABS, CEIL, FLOOR, ROUND).
  - **Fall back to interpreted** for: strings, REGEX, ISIRI, ISBLANK, mixed types.
  - **Files:** `src/jit/{mod.rs,filter_compiler.rs,jit_cache.rs}`, `src/lib.rs`, `engine/oxirs-arq/Cargo.toml`, `Cargo.toml` (workspace deps), `tests/jit_test.rs`, `benches/jit_bench.rs`.
- [x] JIT phase c — hash-join key comparison + ORDER BY codegen (planned 2026-05-01)
  - **Goal:** Extend Cranelift JIT to two more operator families: (1) join key equality comparison for hash-join probe phase; (2) ORDER BY multi-column comparator. Both produce native functions via JITModule, falling back to interpreted for non-numeric keys.
  - **Files:** `src/jit/join_compiler.rs`, `src/jit/order_compiler.rs`, `src/jit/mod.rs` (extended), `src/lib.rs`, `tests/jit_test.rs` (extended).
  - **JoinCompiler:** signature `fn(*const f64, *const f64, usize) -> i8`; per-key epsilon (`|a-b| < 1e-9`) and exact (bitcast to i64 + icmp Equal) modes; chained with `band`. ZST pattern (fresh JITModule per compile call).
  - **OrderCompiler:** signature `fn(*const f64, *const f64, usize) -> i8` returning -1/0/1; select-chain IR (no control-flow blocks); per-column ascending/descending flags baked in.
  - **Tests:** 21 new tests in `tests/jit_test.rs` covering match/no-match, multi-key, epsilon/exact/NaN semantics, ascending/descending, multi-column short-circuit, Ordering enum correctness.
  - **Result:** 2886 tests passing (all); 41 JIT-specific tests (21 new); zero clippy warnings in both `--features jit` and default feature configurations.
- [x] JIT phase d — PROJECT/DISTINCT/HAVING codegen (completed 2026-05-01)
  - **Goal:** Final phase completing SPARQL operator JIT coverage. Three new operator families: (1) ProjectCompiler — column extraction/reorder for PROJECT and GROUP BY; (2) DistinctCompiler — FNV-1a hash over key columns for DISTINCT deduplication; (3) HavingCompiler — thin wrapper over FilterCompiler for HAVING clause predicates over aggregate results.
  - **Files:** `src/jit/{project_compiler,distinct_compiler,having_compiler}.rs`, `src/jit/mod.rs`, `src/lib.rs`, `tests/jit_test.rs`.

## Jena Parity Gaps (identified 2026-05-01)

- [x] JenaText SPARQL integration — text:query property function backed by tantivy full-text index. SPARQL queries can use `?doc text:query (text:label "search term")` as a triple pattern that dispatches to tantivy. Implemented in `engine/oxirs-arq/src/text_search/` (index.rs + property_fn.rs + mod.rs) behind the `text-search` feature flag. 15 integration tests in `tests/text_search_test.rs`; zero clippy warnings in both `--features text-search` and default configurations.
- [x] Jena Rule Language (.rules) parser — parse Jena's own rule syntax alongside the existing Datalog dialect. Required for Jena config migration compatibility. Implemented in `engine/oxirs-rule/src/jena_rl/` (lexer.rs + parser.rs + lowering.rs + mod.rs). Handles forward rules (`body -> head`), backward rules (`head <- body`), `@prefix` declarations, variables (`?x`), full IRIs, prefixed names, string/int/float literals, and built-in atoms (`notEqual`→`RuleAtom::NotEqual`, `lessThan`→`RuleAtom::LessThan`, `greaterThan`→`RuleAtom::GreaterThan`, others→`RuleAtom::Builtin`). Default prefixes (`rdf`, `rdfs`, `xsd`, `owl`) are pre-populated. Public API: `parse_jrl()` and `parse_and_lower()`. 27 tests across unit + integration (`tests/jena_rl_test.rs`); zero clippy warnings.
