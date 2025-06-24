# OxiRS Core TODO

## Current Status: Foundation Implementation (Phase 0)

### Core Data Model Implementation

#### RDF Terms (Priority: Critical)
- [ ] **NamedNode implementation**
  - [ ] IRI validation according to RFC 3987
  - [ ] Efficient string storage (Cow<str> or Arc<str>)
  - [ ] Display and Debug traits
  - [ ] Hash and Eq implementations
  - [ ] Serialization support (serde)

- [ ] **BlankNode implementation**
  - [ ] Scoped identifier generation
  - [ ] Thread-safe ID allocation
  - [ ] Consistent serialization across sessions
  - [ ] Collision detection and resolution

- [ ] **Literal implementation**
  - [ ] XSD datatype support (string, integer, decimal, boolean, etc.)
  - [ ] Language tag validation (BCP 47)
  - [ ] Custom datatype registration
  - [ ] Value extraction and comparison
  - [ ] Canonical form normalization

- [ ] **Variable implementation**
  - [ ] SPARQL variable naming rules
  - [ ] Scoping for nested queries
  - [ ] Binding mechanism

#### Graph Structures (Priority: Critical)
- [ ] **Triple implementation**
  - [ ] Memory-efficient storage
  - [ ] Pattern matching support
  - [ ] Ordering for btree indexes
  - [ ] Serialization formats

- [ ] **Quad implementation**
  - [ ] Named graph context handling
  - [ ] Default graph semantics
  - [ ] Union graph operations

- [ ] **Graph container**
  - [ ] HashSet-based implementation for uniqueness
  - [ ] Iterator interface for traversal
  - [ ] Bulk insert/remove operations
  - [ ] Memory usage optimization

- [ ] **Dataset container**
  - [ ] Named graph management
  - [ ] Default graph handling
  - [ ] Cross-graph queries
  - [ ] SPARQL dataset semantics

### Parser/Serializer Framework (Priority: High)

#### Core Infrastructure
- [ ] **Format detection**
  - [ ] MIME type mapping
  - [ ] File extension detection
  - [ ] Content sniffing for ambiguous cases
  - [ ] Registry for custom formats

- [ ] **Streaming interfaces**
  - [ ] AsyncRead/AsyncWrite support
  - [ ] Incremental parsing for large files
  - [ ] Error recovery mechanisms
  - [ ] Progress reporting

#### Format Support (Port from Oxigraph)
- [ ] **Turtle format** (oxttl port)
  - [ ] Complete Turtle 1.1 grammar
  - [ ] Prefix handling and expansion
  - [ ] Base IRI resolution
  - [ ] Pretty-printing serializer

- [ ] **N-Triples format**
  - [ ] Streaming line-by-line parser
  - [ ] Minimal memory footprint
  - [ ] Error line reporting

- [ ] **TriG format**
  - [ ] Named graph syntax
  - [ ] Turtle compatibility mode
  - [ ] Graph label validation

- [ ] **N-Quads format**
  - [ ] Quad-based streaming
  - [ ] Default graph handling
  - [ ] Validation and normalization

- [ ] **RDF/XML format** (oxrdfxml port)
  - [ ] XML namespaces handling
  - [ ] RDF/XML abbreviations
  - [ ] DOM-free streaming parser
  - [ ] XML canonicalization

- [ ] **JSON-LD format** (oxjsonld port)
  - [ ] Context processing and caching
  - [ ] Expansion and compaction algorithms
  - [ ] Frame support
  - [ ] Remote context loading

### Integration Layer (Priority: High)

#### Oxigraph Compatibility
- [ ] **Direct integration**
  - [ ] Convert between oxirs and oxigraph types
  - [ ] Performance benchmarking vs oxigraph
  - [ ] Memory usage comparison
  - [ ] API compatibility layer

- [ ] **Testing suite**
  - [ ] Round-trip serialization tests
  - [ ] Compatibility with oxigraph test cases
  - [ ] Performance regression tests

#### Error Handling
- [ ] **Comprehensive error types**
  - [ ] Parse errors with position information
  - [ ] Validation errors with context
  - [ ] I/O errors with retry policies
  - [ ] Network errors for remote resources

- [ ] **Error recovery**
  - [ ] Graceful handling of malformed data
  - [ ] Partial parsing success
  - [ ] Warning collection for non-fatal issues

### Performance Optimization (Priority: Medium)

#### Memory Management
- [ ] **String interning**
  - [ ] Global IRI interning
  - [ ] Datatype IRI deduplication
  - [ ] Memory pool for temporary strings

- [ ] **Zero-copy operations**
  - [ ] Cow<str> for owned/borrowed strings
  - [ ] View types for graph subsets
  - [ ] Lazy evaluation for expensive operations

#### Concurrent Access
- [ ] **Thread safety**
  - [ ] Arc/Mutex for shared graphs
  - [ ] Lock-free data structures where possible
  - [ ] Reader-writer locks for graphs

- [ ] **Parallel processing**
  - [ ] Parallel parsing for large files
  - [ ] Concurrent graph operations
  - [ ] Rayon integration for iterators

### Documentation & Testing (Priority: Medium)

#### Documentation
- [ ] **API documentation**
  - [ ] Comprehensive rustdoc comments
  - [ ] Usage examples for all major types
  - [ ] Integration guides
  - [ ] Performance characteristics

- [ ] **Tutorials**
  - [ ] Getting started guide
  - [ ] Common patterns and idioms
  - [ ] Integration with other crates

#### Testing
- [ ] **Unit tests**
  - [ ] 100% code coverage for core types
  - [ ] Edge case handling
  - [ ] Error condition testing

- [ ] **Integration tests**
  - [ ] Cross-format serialization
  - [ ] Large dataset handling
  - [ ] Performance benchmarks

- [ ] **Compliance tests**
  - [ ] W3C RDF test suite
  - [ ] Format-specific conformance tests
  - [ ] Interoperability with other libraries

## Phase 1 Dependencies

### Required for SPARQL Engine
- [ ] Variable binding interface
- [ ] Graph pattern matching
- [ ] Result set construction

### Required for GraphQL Layer  
- [ ] Type introspection
- [ ] Schema generation helpers
- [ ] Resolver compatibility

### Required for AI Integration
- [ ] Vector embedding support
- [ ] Similarity computation
- [ ] Clustering interfaces

## Estimated Timeline

- **Core data model**: 4-6 weeks
- **Parser framework**: 6-8 weeks  
- **Format implementations**: 8-10 weeks
- **Integration & testing**: 4-6 weeks
- **Performance optimization**: 4-6 weeks

**Total estimate**: 26-36 weeks (Phase 0 completion)