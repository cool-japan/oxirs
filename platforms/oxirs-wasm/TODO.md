# oxirs-wasm TODO

## High Priority

- [ ] **Persistent Storage**:
  - [ ] IndexedDB backend for browsers
  - [ ] localStorage fallback for small datasets
  - [ ] Automatic sync between memory and storage
  - [ ] Transaction support

- [ ] **Full SPARQL Support**:
  - [ ] SPARQL UPDATE (INSERT, DELETE)
  - [ ] SPARQL DESCRIBE queries
  - [ ] Property paths in patterns
  - [ ] FILTER expressions (beyond basic)
  - [ ] OPTIONAL patterns
  - [ ] UNION, MINUS operations
  - [ ] Aggregates (COUNT, SUM, AVG, etc.)
  - [ ] GROUP BY, HAVING
  - [ ] ORDER BY

- [ ] **Advanced Parsing**:
  - [ ] JSON-LD support
  - [ ] RDF/XML parsing
  - [ ] TriG (named graphs)
  - [ ] N-Quads support

## Medium Priority

- [ ] **Reasoning Support**:
  - [ ] RDFS inference (subClassOf, subPropertyOf)
  - [ ] OWL reasoning (basic subset)
  - [ ] Rule-based inference
  - [ ] Integration with oxirs-rule (WASM-compatible subset)

- [ ] **SHACL Validation**:
  - [ ] Port oxirs-shacl core to WASM
  - [ ] Client-side data validation
  - [ ] Real-time validation feedback

- [ ] **Streaming Results**:
  - [ ] Async iterators for large result sets
  - [ ] Chunked query execution
  - [ ] Cancellable queries

- [ ] **RDF-star Support**:
  - [ ] Parse RDF-star syntax
  - [ ] Store quoted triples
  - [ ] Query quoted triples

## Low Priority

- [ ] **Optimization**:
  - [ ] SIMD for triple matching (wasm-simd128)
  - [ ] Parallel query execution (web workers)
  - [ ] Query optimization (join reordering)
  - [ ] Index selection strategies

- [ ] **Federation**:
  - [ ] SERVICE keyword for remote SPARQL
  - [ ] Fetch remote graphs
  - [ ] Distributed query execution

- [ ] **Import/Export**:
  - [ ] Binary serialization (faster loading)
  - [ ] Compressed formats (gzip)
  - [ ] Streaming import for large files

## API Enhancements

- [ ] **Query Builder API**:
  ```typescript
  store.select('?s', '?o')
       .where('?s', 'foaf:knows', '?o')
       .filter('?s != ?o')
       .limit(10)
       .execute()
  ```

- [ ] **Reactive Queries**:
  ```typescript
  const observable = store.observe('SELECT * WHERE { ?s ?p ?o }');
  observable.subscribe(results => {
      console.log('Results updated:', results);
  });
  ```

- [ ] **Transactions**:
  ```typescript
  const tx = store.transaction();
  tx.insert(...);
  tx.delete(...);
  await tx.commit();
  ```

## Browser Integration

- [ ] **Service Worker**:
  - Background sync for offline support
  - Cache-first loading strategies
  - Progressive Web App (PWA) support

- [ ] **Web Components**:
  - `<oxirs-sparql-editor>` custom element
  - `<oxirs-rdf-visualizer>` graph visualization
  - `<oxirs-query-builder>` GUI query builder

- [ ] **Framework Integration**:
  - React hooks (`useOxiRSStore`, `useSPARQL`)
  - Vue composables
  - Svelte stores

## Performance

- [ ] **Memory Management**:
  - Implement LRU eviction for large datasets
  - Configurable memory limits
  - Memory pressure detection (browser API)

- [ ] **Lazy Loading**:
  - Load triples on-demand from IndexedDB
  - Virtual scrolling for large result sets
  - Pagination at storage level

- [ ] **Worker Threads**:
  - Offload parsing to web workers
  - Parallel query execution
  - Background indexing

## Testing

- [ ] Browser testing with wasm-pack test
- [ ] Integration tests with headless Chrome
- [ ] Performance benchmarks in browser
- [ ] Memory leak detection
- [ ] Bundle size optimization tests

## Tooling

- [ ] NPM package publishing automation
- [ ] GitHub Actions for WASM build
- [ ] CDN deployment (unpkg, jsdelivr)
- [ ] Webpack/Vite/Rollup plugins

## Documentation

- [ ] Interactive tutorial with RunKit/CodePen
- [ ] API reference with examples
- [ ] Performance guide
- [ ] Browser compatibility matrix
- [ ] Migration guide from other RDF libraries

## Comparison & Benchmarks

Compare against:
- [ ] rdflib.js (JavaScript RDF library)
- [ ] N3.js (Turtle parser)
- [ ] Comunica (federated SPARQL)
- [ ] Graphy (RDF toolkit)

## Ecosystem Integration

- [ ] Solid Pod integration (read/write)
- [ ] RDF browser extensions
- [ ] Observable notebooks
- [ ] Jupyter kernels (via Deno)

## Future Ideas

- [ ] WebGPU acceleration for graph algorithms
- [ ] WebRTC for P2P RDF sync
- [ ] IPFS integration for decentralized storage
- [ ] WebAuthn for signing operations
- [ ] Credential Handler API integration

## Dependencies to Consider

- `web-sys` features: IndexedDB, LocalStorage, ServiceWorker
- `gloo` - Rust/WASM convenience library
- `yew` / `leptos` - For example applications
