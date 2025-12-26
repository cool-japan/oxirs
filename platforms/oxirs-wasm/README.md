# oxirs-wasm

**WebAssembly bindings for OxiRS - Run RDF/SPARQL in the browser**

[![Crates.io](https://img.shields.io/crates/v/oxirs-wasm.svg)](https://crates.io/crates/oxirs-wasm)
[![docs.rs](https://docs.rs/oxirs-wasm/badge.svg)](https://docs.rs/oxirs-wasm)
[![npm](https://img.shields.io/npm/v/oxirs-wasm.svg)](https://www.npmjs.com/package/oxirs-wasm)

Lightweight RDF and SPARQL implementation compiled to WebAssembly for browser, Node.js, and edge deployment.

## Features

- **In-memory RDF Store**: HashSet-based triple storage with O(1) lookups
- **RDF Parsing**: Turtle and N-Triples format support
- **SPARQL Queries**: SELECT, ASK, CONSTRUCT with pattern matching
- **TypeScript Support**: Full type definitions
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
oxirs-wasm = "0.1"
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

// Execute SPARQL query
const results = await store.query(`
    SELECT ?person ?name WHERE {
        ?person :name ?name .
    }
`);
console.log(results);
// [{ person: "http://example.org/bob", name: "\"Bob\"" }]

// ASK query
const exists = await store.ask(`
    ASK { :alice :knows :bob }
`);
console.log('Alice knows Bob:', exists); // true

// Export to N-Triples
const ntriples = store.toNTriples();
console.log(ntriples);
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
        import { initialize, createStore } from './pkg/oxirs_wasm.js';

        async function run() {
            await initialize();
            const store = await createStore();

            await store.loadTurtle(`
                @prefix : <http://example.org/> .
                :alice :knows :bob .
            `);

            const results = await store.query(
                'SELECT ?s ?o WHERE { ?s :knows ?o }'
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

    // SPARQL queries
    query(sparql: string): Promise<QueryResult[]>;
    ask(sparql: string): Promise<boolean>;
    construct(sparql: string): Promise<Triple[]>;

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
    import init, { createStore } from './pkg/oxirs_wasm.js';
    await init();
    const store = await createStore();
</script>
```

### Node.js

```javascript
const { initialize, createStore } = require('oxirs-wasm');

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

Current implementation focuses on core functionality:

- ❌ No persistent storage (in-memory only)
- ❌ No SPARQL UPDATE/DESCRIBE
- ❌ No RDFS/OWL reasoning
- ❌ No SHACL validation
- ❌ No federated queries (SERVICE)
- ❌ Limited SPARQL features (basic graph patterns only)

For full SPARQL support, use the server-side `oxirs-fuseki` crate.

## Performance

- **WASM binary size**: ~300KB (optimized with wasm-opt)
- **Initialization**: ~100ms (first load, cached)
- **Parsing**: 10K triples/sec (Turtle)
- **Query execution**: 1K queries/sec (simple patterns)
- **Memory**: ~200KB per 1K triples

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
- `getrandom` - Random number generation (js feature)

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
