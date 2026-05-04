# OxiRS WASM - TODO

*Version: 0.3.0 | Last Updated: May 3, 2026*

## Status: Production Ready

OxiRS WASM v0.2.3 provides WebAssembly bindings for browser-based RDF and SPARQL processing, enabling semantic web applications directly in the browser.

### Features
- ✅ WASM compilation with wasm-bindgen
- ✅ JavaScript/TypeScript API bindings
- ✅ RDF Triple and Graph stores
- ✅ Full SPARQL 1.1 query support (OPTIONAL, UNION, FILTER)
- ✅ SPARQL UPDATE operations (INSERT, DELETE)
- ✅ Property paths in patterns
- ✅ Aggregates (COUNT, SUM, AVG, GROUP BY)
- ✅ Turtle, N-Triples, N-Quads, TriG parsing
- ✅ Named graphs support
- ✅ Memory-efficient graph operations
- ✅ Browser compatibility (Chrome, Firefox, Safari, Edge)
- ✅ Query builder for composable SPARQL
- ✅ Triple store, storage adapter
- ✅ Namespace manager, endpoint client
- ✅ WASM bridge, geojson support
- ✅ Property path evaluator
- ✅ 858 tests passing

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ WASM compilation, JS/TS bindings, basic SPARQL, Turtle/N-Triples

### v0.2.3 - Current Release (March 16, 2026)
- ✅ Full SPARQL 1.1 query support
- ✅ SPARQL UPDATE operations
- ✅ Property paths in patterns
- ✅ Aggregates (COUNT, SUM, AVG, GROUP BY)
- ✅ Named graphs support (TriG, N-Quads)
- ✅ Query builder, triple store, storage adapter
- ✅ Namespace manager, endpoint client, geojson support
- ✅ 858 tests passing

### v0.3.0 - Planned (Q2 2026)
- [~] Long-term support guarantees (policy: docs/policies/lts.md)
- [x] React/Vue/Svelte integration hooks (completed 2026-04-28)
  - **Goal:** Thin idiomatic TypeScript adapters wrapping existing wasm-bindgen API for each JS framework.
  - **Design:** @oxirs/react (hooks), @oxirs/vue (composables), @oxirs/svelte (stores). Each ≤200 LoC TypeScript. Generated alongside existing JS distribution.
  - **Files:** js/react/index.ts (new), js/vue/index.ts (new), js/svelte/index.ts (new)
  - **Tests:** TypeScript structural validation; Rust integration tests for underlying API completeness
- [x] Web Workers for parallel execution (planned 2026-04-17)
  - **Goal:** Add Web Worker pool enabling parallel SPARQL query execution off the main thread
  - **Design:** WorkerPool creates N web_sys::Worker instances; query dispatch via postMessage; result collection via Promise/channel pattern; avoid SharedArrayBuffer to sidestep COOP/COEP requirement; use MessageChannel for bidirectional comms
  - **Files:** src/web_worker.rs (new), src/lib.rs
  - **Tests:** Worker pool dispatches 10 parallel queries; result ordering correctness; error propagation from worker
  - **Risk:** SharedArrayBuffer requires COOP/COEP headers; design around MessageChannel instead
- [x] ServiceWorker for offline support (planned 2026-04-17)
  - **Goal:** Add ServiceWorker with IndexedDB-backed offline cache for RDF data and SPARQL query results
  - **Design:** OfflineCache struct using IndexedDB (web_sys::IdbFactory); cache query→result pairs with configurable TTL; sync_on_online() deferred upload when connectivity restored; cache invalidation on store mutation
  - **Files:** src/offline_cache.rs (new), src/lib.rs
  - **Tests:** Cache hit/miss correctness; TTL expiry; sync trigger on online event; serialization round-trip
  - **Risk:** IndexedDB API is async; use wasm-bindgen-futures for bridge
- [x] RDFS inference (completed 2026-04-28)
  - **Goal:** RDFS entailment rules as a forward-chaining fixed-point module exported to JS.
  - **Design:** Semi-naive forward chaining (subClassOf, subPropertyOf, domain, range, type propagation). Export inferRdfs() -> {added: number}.
  - **Files:** src/inference/mod.rs (existing), store/mod.rs +inferRdfs() wasm export, tests/rdfs_inference_integration.rs (new)
  - **Status:** Core rules (rdfs2/3/5/7/9/11) already implemented in src/inference/mod.rs; WASM export added as inferRdfs() on OxiRSStore
- [x] SHACL validation (subset) (planned 2026-04-17)
  - **Goal:** Implement SHACL core subset validator for NodeShape and PropertyShape constraints in WASM
  - **Design:** ShaclValidator parses Turtle shape graphs; validates sh:NodeShape and sh:PropertyShape; constraints: sh:minCount, sh:maxCount, sh:datatype, sh:pattern, sh:minInclusive, sh:maxInclusive, sh:class, sh:nodeKind; returns ValidationReport with per-constraint violation messages; wasm_bindgen exposed API
  - **Files:** src/shacl/mod.rs (new), src/shacl/shapes.rs (new), src/shacl/validator.rs (new), src/lib.rs
  - **Tests:** minCount/maxCount violation detection; datatype mismatch report; pattern regex failure; valid shape graph passes without violations
  - **Risk:** SHACL subset only — explicitly document which constraints are and are not supported

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS WASM v0.2.3 - Semantic web in the browser*
