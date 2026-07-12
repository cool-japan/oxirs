# oxirs-wasm

**WebAssembly bindings for OxiRS - Run RDF/SPARQL in the browser**

[![Crates.io](https://img.shields.io/crates/v/oxirs-wasm.svg)](https://crates.io/crates/oxirs-wasm)
[![docs.rs](https://docs.rs/oxirs-wasm/badge.svg)](https://docs.rs/oxirs-wasm)
[![npm](https://img.shields.io/npm/v/oxirs-wasm.svg)](https://www.npmjs.com/package/oxirs-wasm)

Lightweight RDF and SPARQL implementation compiled to WebAssembly for browser, Node.js, and edge deployment.

## Features

- **In-memory RDF Store**: Subject/predicate/object hash-indexed triple storage; pattern and property-path evaluation looks up matches in the index instead of scanning every triple in the store
- **RDF Parsing**: Turtle, N-Triples, N-Quads, TriG, plus a streaming/incremental chunk-based parser
- **SPARQL 1.1 Queries**: SELECT, ASK, CONSTRUCT with OPTIONAL, UNION, FILTER (LANG/STR/DATATYPE/BOUND/regex/isIRI/isLiteral/isBlank/STRSTARTS/STRENDS/CONTAINS/STRLEN, `&&`/`||`/`!`), FILTER EXISTS/NOT EXISTS, property paths (`/ | * + ? ^ !()`), subqueries, GROUP BY + aggregates, ORDER BY, LIMIT/OFFSET
- **SPARQL UPDATE**: INSERT DATA, DELETE DATA, INSERT/DELETE ... WHERE, CLEAR, DROP
- **PREFIX/BASE Prologues**: queries may declare `PREFIX`/`BASE` and use prefixed names, expanded before parsing
- **Solution Budget**: cap intermediate join solutions per query so an unselective join fails fast instead of running to completion
- **Named Graphs**: multi-graph (quad) storage with `GRAPH` pattern queries
- **RDFS Inference**: forward-chaining subClassOf/subPropertyOf/domain/range entailment
- **SHACL Validation (core subset)**: NodeShape/PropertyShape constraint checking
- **TypeScript Support**: type definitions, plus React/Vue/Svelte adapters (`js/react`, `js/vue`, `js/svelte`)
- **Zero Server Dependencies**: No Tokio, runs in single-threaded WASM
- **Lightweight**: ~300KB optimized binary

## Installation

### NPM (Browser/Node.js)

```bash
npm install oxirs-wasm
```

### Cargo (Rust WASM Project)

```toml
[dependencies]
oxirs-wasm = "0.3.2"
```

## Quick Start

### JavaScript/TypeScript

```typescript
import { initialize, createStore } from 'oxirs-wasm';

// Initialize WASM module
await initialize();

// Create RDF store
const store = await createStore();

// Load Turtle data
const turtle = `
    @prefix : <http://example.org/> .
    :alice :knows :bob .
    :bob :name "Bob" .
`;
const count = await store.loadTurtle(turtle);
console.log(`Loaded ${count} triples`);

// Execute SPARQL query — PREFIX/BASE prologues are expanded before parsing
const results = await store.query(`
    PREFIX : <http://example.org/>
    SELECT ?person ?name WHERE {
        ?person :name ?name .
    }
`);
console.log(results);
// [{ person: "http://example.org/bob", name: "\"Bob\"" }]

// ASK query
const exists = await store.ask(`
    PREFIX : <http://example.org/>
    ASK { :alice :knows :bob }
`);
console.log('Alice knows Bob:', exists); // true

// Export to N-Triples
const ntriples = store.toNTriples();
console.log(ntriples);
```

### Bounding Query Cost

A join is evaluated left to right, so an unselective pattern early in a
`WHERE` clause can build a huge intermediate result before a later pattern
cuts it back down — `LIMIT` cannot bound this because it only applies at the
end. `setSolutionBudget` caps the number of intermediate solutions a single
query may produce, so a runaway join fails fast with a query error instead of
running to completion:

```typescript
// Cap intermediate solutions (useful when answering queries under a
// time/CPU budget, e.g. a serverless endpoint).
store.setSolutionBudget(50_000);

try {
    await store.query('SELECT * WHERE { ?a ?p ?x . ?b ?p ?y }');
} catch (e) {
    console.error('Query exceeded its solution budget:', e);
}

// Remove the cap — queries go back to being unbounded.
store.clearSolutionBudget();
```

### HTML Integration

```html
<!DOCTYPE html>
<html>
<head>
    <title>RDF in Browser</title>
</head>
<body>
    <script type="module">
        // Loading the raw wasm-pack output directly (no npm wrapper): use
        // the generated `init` loader and the `OxiRSStore` constructor.
        import init, { OxiRSStore } from './pkg/oxirs_wasm.js';

        async function run() {
            await init();
            const store = new OxiRSStore();

            await store.loadTurtle(`
                @prefix : <http://example.org/> .
                :alice :knows :bob .
            `);

            const results = await store.query(
                'PREFIX : <http://example.org/>\nSELECT ?s ?o WHERE { ?s :knows ?o }'
            );

            console.log('Results:', results);
        }

        run();
    </script>
</body>
</html>
```

## Building from Source

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build WASM package
cd platforms/oxirs-wasm
wasm-pack build --target web --release

# The output will be in pkg/
# - oxirs_wasm.js (ES module)
# - oxirs_wasm_bg.wasm (binary)
# - oxirs_wasm.d.ts (TypeScript types)
```

## API Reference

### Store Operations

```typescript
class OxiRSStore {
    constructor();

    // Load RDF data
    loadTurtle(turtle: string): Promise<number>;
    loadNTriples(ntriples: string): Promise<number>;

    // Triple operations
    insert(subject: string, predicate: string, object: string): boolean;
    delete(subject: string, predicate: string, object: string): boolean;
    contains(subject: string, predicate: string, object: string): boolean;
    size(): number;
    clear(): void;

    // SPARQL queries (PREFIX/BASE prologues are supported)
    query(sparql: string): Promise<QueryResult[]>;
    ask(sparql: string): Promise<boolean>;
    construct(sparql: string): Promise<Triple[]>;

    // Query cost control
    setSolutionBudget(budget: number): void;
    clearSolutionBudget(): void;

    // RDFS forward-chaining inference
    inferRdfs(): { added: number };

    // Export
    toTurtle(): string;
    toNTriples(): string;

    // Indexes
    subjects(): string[];
    predicates(): string[];
    objects(): string[];

    // Namespaces
    addPrefix(prefix: string, uri: string): void;
}
```

## Deployment Targets

### Browser

```html
<script type="module">
    import init, { OxiRSStore } from './pkg/oxirs_wasm.js';
    await init();
    const store = new OxiRSStore();
</script>
```

### Node.js

The `web` build target produces an ES module, so import it rather than
`require()` it (Node.js cannot `require()` an ES module):

```javascript
import { initialize, createStore } from 'oxirs-wasm';

(async () => {
    await initialize();
    const store = await createStore();
    // ...
})();
```

### Cloudflare Workers

```typescript
import { initialize, createStore } from 'oxirs-wasm';

export default {
    async fetch(request: Request): Promise<Response> {
        await initialize();
        const store = await createStore();

        // Process RDF data
        const { rdf, sparql } = await request.json();
        await store.loadTurtle(rdf);
        const results = await store.query(sparql);

        return new Response(JSON.stringify(results));
    }
};
```

### Deno Deploy

```typescript
import { initialize, createStore } from 'https://esm.sh/oxirs-wasm';

Deno.serve(async (req) => {
    await initialize();
    const store = await createStore();
    // ...
});
```

## Limitations

- ❌ No persistent storage (in-memory only)
- ❌ No SPARQL DESCRIBE
- ❌ No OWL reasoning (RDFS forward-chaining entailment — subClassOf/subPropertyOf/domain/range — is supported via `inferRdfs()`)
- ❌ SHACL validation covers a core subset only (NodeShape/PropertyShape with minCount, maxCount, datatype, pattern, minInclusive/maxInclusive, class, nodeKind, in, hasValue — not the full SHACL-AF/SPARQL-based constraint set)
- ❌ No federated queries (SPARQL `SERVICE`)

For full SPARQL 1.1 and complete SHACL support, use the server-side `oxirs-fuseki` / `oxirs-shacl` crates.

## Performance

- **WASM binary size**: ~300KB (optimized with wasm-opt)
- **Initialization**: ~100ms (first load, cached)
- **Parsing**: 10K triples/sec (Turtle)
- **Query execution**: 1K queries/sec (simple patterns)
- **Memory**: ~200KB per 1K triples
- **Indexed evaluation**: triple pattern and property-path matching go through the store's subject/predicate/object hash indexes rather than scanning every triple, so a join costs a hash lookup per solution instead of a full graph scan

## Use Cases

- **Browser RDF Editors**: Client-side validation and editing
- **Offline Applications**: Local knowledge graphs without server
- **Edge Computing**: IoT devices, embedded systems
- **Privacy-Preserving**: Process sensitive data locally
- **Developer Tools**: RDF/SPARQL playground in browser
- **Static Sites**: Jamstack with client-side queries

## Dependencies

- `wasm-bindgen` - JavaScript interop
- `js-sys` - JavaScript standard library bindings
- `web-sys` - Web APIs (console, performance)
- `console_error_panic_hook` - Readable panic messages in the browser console
- `serde`, `serde_json` - Serialization
- `thiserror` - Error types

## License

Licensed under the Apache License, Version 2.0.
