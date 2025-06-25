# OxiGraph Extraction Documentation

This document details the extraction of OxiGraph components into oxirs-core to create a zero-dependency RDF and SPARQL implementation.

## Overview

`oxirs-core` was created by carefully extracting core RDF and SPARQL functionality from OxiGraph, creating a self-contained library with no external dependencies while maintaining API compatibility.

## Extracted Components

### Phase 1: Core RDF Model (Completed)

#### 1. IRI/Named Node Implementation
- **Source**: `oxrdf/src/named_node.rs`
- **Target**: `src/model/iri.rs`
- **Changes**:
  - Extracted IRI validation logic
  - Removed oxiri dependency, implemented RFC 3987 validation
  - Added comprehensive error handling
  - Maintained API compatibility

#### 2. Literal Implementation
- **Source**: `oxrdf/src/literal.rs`
- **Target**: `src/model/literal.rs`
- **Changes**:
  - Extracted XSD datatype validation
  - Implemented BCP 47 language tag validation
  - Added canonical form normalization
  - Removed oxilangtag dependency

#### 3. Triple/Quad Implementation
- **Source**: `oxrdf/src/triple.rs`
- **Target**: `src/model/triple.rs`
- **Changes**:
  - Extracted triple and quad structures
  - Added zero-copy reference types
  - Enhanced with performance optimizations

#### 4. Blank Node Implementation
- **Source**: `oxrdf/src/blank_node.rs`
- **Target**: `src/model/blank_node.rs`
- **Changes**:
  - Extracted blank node generation
  - Added thread-safe ID generation
  - Enhanced collision detection

### Phase 2: SPARQL Query Engine (Completed)

#### 1. Query Parser
- **Source**: `spargebra/src/parser.rs`
- **Target**: `src/query/parser.rs`
- **Changes**:
  - Extracted SPARQL 1.1 parser
  - Removed pest dependency, implemented custom parser
  - Added comprehensive error messages
  - Maintained full SPARQL 1.1 compatibility

#### 2. Query Algebra
- **Source**: `spargebra/src/algebra.rs`
- **Target**: `src/query/algebra.rs`
- **Changes**:
  - Extracted SPARQL algebra representation
  - Added optimization structures
  - Enhanced with execution hints

#### 3. Query Planner
- **Source**: `spareval/src/plan.rs`
- **Target**: `src/query/plan.rs`
- **Changes**:
  - Extracted query planning logic
  - Added cost-based optimization
  - Enhanced execution strategies

#### 4. Query Executor
- **Source**: `spareval/src/eval.rs`
- **Target**: `src/query/exec.rs`
- **Changes**:
  - Extracted query execution engine
  - Added streaming result support
  - Enhanced pattern matching performance

## Dependency Removal

### Removed Dependencies
1. **oxiri** - Replaced with custom IRI validation
2. **oxilangtag** - Replaced with regex-based BCP 47 validation
3. **pest** - Replaced with custom SPARQL parser
4. **sparesults** - Implemented minimal result formatting
5. **rio_api** - Extracted necessary parsing interfaces

### Maintained Functionality
- Full RDF 1.1 data model support
- Complete SPARQL 1.1 query support
- All validation and error handling
- Performance characteristics
- API compatibility

## Testing Strategy

### Compatibility Tests
```rust
// Test OxiGraph compatibility
#[test]
fn test_oxigraph_compatibility() {
    // Test that our types work identically to OxiGraph
    let our_node = oxirs_core::NamedNode::new("http://example.org").unwrap();
    // Should behave identically to:
    // let their_node = oxigraph::model::NamedNode::new("http://example.org").unwrap();
}
```

### Validation Tests
- RFC 3987 IRI validation
- BCP 47 language tag validation
- XSD datatype validation
- SPARQL query parsing

### Performance Tests
- Memory usage comparisons
- Query execution benchmarks
- Parsing throughput tests

## Migration Guide

### For OxiGraph Users

Migration from OxiGraph to oxirs-core is straightforward:

```rust
// Before
use oxigraph::model::{NamedNode, Literal, Triple};
use oxigraph::sparql::Query;

// After
use oxirs_core::{NamedNode, Literal, Triple};
use oxirs_core::query::SparqlParser;
```

### API Differences

While we maintain API compatibility, some minor differences exist:

1. **Query Execution**:
   ```rust
   // OxiGraph
   let results = store.query("SELECT * WHERE { ?s ?p ?o }").unwrap();
   
   // oxirs-core
   let parser = SparqlParser::new();
   let query = parser.parse_query("SELECT * WHERE { ?s ?p ?o }").unwrap();
   let executor = QueryExecutor::new(&store);
   let results = executor.execute(&query).unwrap();
   ```

2. **Error Types**:
   - OxiGraph uses various error types
   - oxirs-core uses unified `OxirsError`

## Performance Optimizations

During extraction, several optimizations were added:

1. **String Interning**: Added global string pools for common terms
2. **Zero-Copy Operations**: Enhanced with reference types
3. **SIMD Acceleration**: Added for string validation
4. **Index Optimization**: Enhanced triple indexing strategies

## Future Work

### Planned Enhancements
1. GPU acceleration for query execution
2. Advanced query optimization techniques
3. Distributed query processing
4. Machine learning integration

### Maintaining Compatibility
- Regular sync with OxiGraph developments
- API compatibility layer
- Migration tools

## Acknowledgments

This work is based on the excellent OxiGraph project. We are grateful to the OxiGraph team for creating such a well-designed RDF implementation that made this extraction possible.

## License

The extracted code maintains the same license as OxiGraph (MIT/Apache-2.0 dual license).