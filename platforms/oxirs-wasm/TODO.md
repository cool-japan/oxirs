# OxiRS WASM - TODO

*Version: 0.3.2 | Last Updated: July 12, 2026*

## Status: Production Ready

OxiRS WASM v0.3.2 provides WebAssembly bindings for browser-based RDF and SPARQL processing, enabling semantic web applications directly in the browser.

### Features
- ✅ WASM compilation with wasm-bindgen
- ✅ JavaScript/TypeScript API bindings (plus React/Vue/Svelte adapters)
- ✅ RDF Triple and Graph stores, indexed on subject/predicate/object
- ✅ Full SPARQL 1.1 query support (OPTIONAL, UNION, FILTER, FILTER EXISTS/NOT EXISTS, subqueries)
- ✅ SPARQL UPDATE operations (INSERT DATA, DELETE DATA, INSERT/DELETE ... WHERE, CLEAR, DROP)
- ✅ `PREFIX`/`BASE` prologue expansion for prefixed names in queries
- ✅ Per-store solution budget — fails a query fast once a join exceeds the configured intermediate-row cap
- ✅ Index-based triple pattern and property-path evaluation (subject/predicate/object hash indexes, not a full scan)
- ✅ Property paths in patterns
- ✅ Aggregates (COUNT, SUM, AVG, GROUP BY)
- ✅ Turtle, N-Triples, N-Quads, TriG parsing
- ✅ Named graphs support
- ✅ RDFS forward-chaining inference (`inferRdfs()`)
- ✅ SHACL validation (core subset: NodeShape/PropertyShape)
- ✅ Memory-efficient graph operations
- ✅ Browser compatibility (Chrome, Firefox, Safari, Edge)
- ✅ Query builder for composable SPARQL
- ✅ Triple store, storage adapter
- ✅ Namespace manager, endpoint client
- ✅ WASM bridge, geojson support
- ✅ Property path evaluator
- ✅ 915 tests passing

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ WASM compilation, JS/TS bindings, basic SPARQL, Turtle/N-Triples

### v0.2.3 - Released (March 16, 2026)
- ✅ Full SPARQL 1.1 query support
- ✅ SPARQL UPDATE operations
- ✅ Property paths in patterns
- ✅ Aggregates (COUNT, SUM, AVG, GROUP BY)
- ✅ Named graphs support (TriG, N-Quads)
- ✅ Query builder, triple store, storage adapter
- ✅ Namespace manager, endpoint client, geojson support
- ✅ 858 tests passing

### v0.3.0 - Released
- [x] Long-term support guarantees (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)
- [x] React/Vue/Svelte integration hooks (completed 2026-04-28)
  - **Goal:** Thin idiomatic TypeScript adapters wrapping existing wasm-bindgen API for each JS framework.
  - **Design:** @oxirs/react (hooks), @oxirs/vue (composables), @oxirs/svelte (stores). Each ≤200 LoC TypeScript. Generated alongside existing JS distribution.
  - **Files:** js/react/index.ts (new), js/vue/index.ts (new), js/svelte/index.ts (new)
  - **Tests:** TypeScript structural validation; Rust integration tests for underlying API completeness
- [x] Web Workers message protocol (planned 2026-04-17)
  - **Goal:** Add a serializable job/result protocol and a job pool for distributing SPARQL query execution to Web Workers
  - **Design:** `WorkerJob`/`WorkerResult` JSON-serializable message types (`sparql_select`, `parse_turtle`, `count_subjects` job kinds) plus a `WorkerPool` that tracks capacity and completed results. Note: this module ships the Rust-side message protocol and job bookkeeping only — spawning real `web_sys::Worker` instances and wiring `postMessage`/`onmessage` is left to caller-supplied JS, not implemented here
  - **Files:** src/web_worker.rs (new), src/lib.rs
  - **Tests:** job/result JSON round-trip; pool capacity clamping; sequential multi-job dispatch; error-result detection
  - **Risk:** SharedArrayBuffer requires COOP/COEP headers; the message format avoids it in favor of structured-clone postMessage, once a caller wires up real Workers
- [x] Offline cache (TTL, in-memory) (planned 2026-04-17)
  - **Goal:** Add a TTL-based cache for RDF data and SPARQL query results, keyed by URL, with a pending-write-back queue for reconnect
  - **Design:** `OfflineCache` struct backed by an in-memory `HashMap` (`CacheEntry` with TTL + ETag); `sync_on_online()` drains queued `PendingSyncRequest`s for replay when connectivity returns. Note: a real IndexedDB/ServiceWorker-backed browser build is not implemented — this module ships the cache/TTL/sync-queue logic only
  - **Files:** src/offline_cache.rs (new), src/lib.rs
  - **Tests:** put/get/evict; TTL expiry; pending-sync queue drain; contains/age checks
  - **Risk:** IndexedDB wiring is future work; keep the current HashMap-only scope documented accurately
- [x] RDFS inference (completed 2026-04-28)
  - **Goal:** RDFS entailment rules as a forward-chaining fixed-point module exported to JS.
  - **Design:** Semi-naive forward chaining (subClassOf, subPropertyOf, domain, range, type propagation). Export inferRdfs() -> {added: number}.
  - **Files:** src/inference/mod.rs (existing), store/mod.rs +inferRdfs() wasm export, tests/rdfs_inference_integration.rs (new)
  - **Status:** Core rules (rdfs2/3/5/7/9/11) already implemented in src/inference/mod.rs; WASM export added as inferRdfs() on OxiRSStore
- [x] SHACL validation (subset) (planned 2026-04-17)
  - **Goal:** Implement SHACL core subset validator for NodeShape and PropertyShape constraints in WASM
  - **Design:** ShaclValidator parses Turtle shape graphs; validates sh:NodeShape and sh:PropertyShape; constraints: sh:minCount, sh:maxCount, sh:datatype, sh:pattern, sh:minInclusive, sh:maxInclusive, sh:class, sh:nodeKind, sh:in, sh:hasValue; returns ValidationReport with per-constraint violation messages; wasm_bindgen exposed API
  - **Files:** src/shacl/mod.rs (new), src/shacl/shapes.rs (new), src/shacl/validator.rs (new), src/lib.rs
  - **Tests:** minCount/maxCount violation detection; datatype mismatch report; pattern regex failure; valid shape graph passes without violations
  - **Risk:** SHACL subset only — explicitly document which constraints are and are not supported

### v0.3.2 - Current Release (July 12, 2026)
- ✅ `PREFIX`/`BASE` SPARQL prologue support (`query::prefix_expand::expand_prologue`), expanded before parsing so queries can use prefixed names instead of only full `<iri>` forms
- ✅ Per-store solution budget (`setSolutionBudget`/`clearSolutionBudget`) — fails a query fast once a join produces more intermediate rows than configured
- ✅ Triple pattern and property-path evaluation now drive off the store's subject/predicate/object indexes instead of scanning every triple, so a join costs a hash lookup per solution rather than a full graph scan
- ✅ Fixed: pattern and property-path matching now treats bracketed `<iri>` and bare `iri` as equal, so a query no longer silently misses matches depending on which form was used; property paths also gained support for the `a` (`rdf:type`) keyword
- ✅ 915 tests passing

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS WASM v0.3.2 - Semantic web in the browser*
