# OxiRS-Star API Reference

[![Documentation](https://docs.rs/oxirs-star/badge.svg)](https://docs.rs/oxirs-star)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

**Comprehensive API reference for OxiRS-Star RDF-star implementation with production-ready examples and advanced usage patterns.**

## Table of Contents

- [Core Types](#core-types)
- [StarStore API](#starstore-api)
- [Parser API](#parser-api)
- [Serializer API](#serializer-api)
- [Query Engine API](#query-engine-api)
- [SPARQL-star Functions](#sparql-star-functions)
- [Reification API](#reification-api)
- [Configuration API](#configuration-api)
- [Error Handling](#error-handling)
- [Advanced Patterns](#advanced-patterns)
- [Performance Optimization](#performance-optimization)

## Core Types

### StarTriple

The fundamental RDF-star triple type supporting quoted triples.

```rust
use oxirs_star::{StarTriple, StarTerm, StarError};

// Basic triple creation
let triple = StarTriple::new(
    StarTerm::iri("http://example.org/alice")?,
    StarTerm::iri("http://foaf.org/knows")?,
    StarTerm::iri("http://example.org/bob")?,
)?;

// Quoted triple creation
let quoted_triple = StarTriple::new(
    StarTerm::iri("http://example.org/person1")?,
    StarTerm::iri("http://example.org/age")?,
    StarTerm::literal("30")?,
)?;

let meta_triple = StarTriple::new(
    StarTerm::quoted_triple(quoted_triple),
    StarTerm::iri("http://example.org/confidence")?,
    StarTerm::typed_literal("0.95", "http://www.w3.org/2001/XMLSchema#decimal")?,
)?;

// Accessors
assert_eq!(triple.subject().as_iri().unwrap(), "http://example.org/alice");
assert_eq!(triple.predicate().as_iri().unwrap(), "http://foaf.org/knows");
assert_eq!(triple.object().as_iri().unwrap(), "http://example.org/bob");

// Type checking
assert!(meta_triple.subject().is_quoted_triple());
assert!(meta_triple.object().is_literal());
```

### StarTerm

Represents RDF-star terms including quoted triples.

```rust
use oxirs_star::StarTerm;

// IRI terms
let iri_term = StarTerm::iri("http://example.org/resource")?;
assert!(iri_term.is_iri());
assert_eq!(iri_term.as_iri().unwrap(), "http://example.org/resource");

// Literal terms
let literal_term = StarTerm::literal("Hello World");
let typed_literal = StarTerm::typed_literal("42", "http://www.w3.org/2001/XMLSchema#integer")?;
let lang_literal = StarTerm::lang_literal("Bonjour", "fr")?;

assert!(literal_term.is_literal());
assert_eq!(literal_term.as_literal().unwrap().value(), "Hello World");
assert_eq!(typed_literal.as_literal().unwrap().datatype().unwrap(), "http://www.w3.org/2001/XMLSchema#integer");

// Blank node terms
let blank_node = StarTerm::blank_node("_:b1");
assert!(blank_node.is_blank_node());

// Quoted triple terms
let base_triple = StarTriple::new(
    StarTerm::iri("http://example.org/s")?,
    StarTerm::iri("http://example.org/p")?,
    StarTerm::iri("http://example.org/o")?,
)?;
let quoted_term = StarTerm::quoted_triple(base_triple);
assert!(quoted_term.is_quoted_triple());

// Equality and hashing
let term1 = StarTerm::iri("http://example.org/same")?;
let term2 = StarTerm::iri("http://example.org/same")?;
assert_eq!(term1, term2);

use std::collections::HashSet;
let mut set = HashSet::new();
set.insert(term1);
assert!(set.contains(&term2));
```

## StarStore API

### Basic Operations

```rust
use oxirs_star::{StarStore, StarTriple, StarTerm};

let mut store = StarStore::new();

// Insert triples
let triple = StarTriple::new(
    StarTerm::iri("http://example.org/alice")?,
    StarTerm::iri("http://foaf.org/name")?,
    StarTerm::literal("Alice Smith"),
)?;

store.insert(&triple)?;
assert_eq!(store.size(), 1);

// Check containment
assert!(store.contains(&triple)?);

// Remove triples
store.remove(&triple)?;
assert_eq!(store.size(), 0);

// Bulk operations
let triples = vec![
    StarTriple::new(
        StarTerm::iri("http://example.org/alice")?,
        StarTerm::iri("http://foaf.org/knows")?,
        StarTerm::iri("http://example.org/bob")?,
    )?,
    StarTriple::new(
        StarTerm::iri("http://example.org/bob")?,
        StarTerm::iri("http://foaf.org/knows")?,
        StarTerm::iri("http://example.org/charlie")?,
    )?,
];

store.insert_batch(&triples)?;
assert_eq!(store.size(), 2);
```

### Pattern Matching

```rust
use oxirs_star::{StarStore, StarPattern, StarTerm};

let mut store = StarStore::new();
// ... populate store ...

// Match all triples
let all_triples: Vec<StarTriple> = store.match_pattern(&StarPattern::any())?;

// Match specific subject
let alice_triples = store.match_pattern(&StarPattern::new(
    Some(StarTerm::iri("http://example.org/alice")?),
    None,
    None,
))?;

// Match specific predicate
let name_triples = store.match_pattern(&StarPattern::new(
    None,
    Some(StarTerm::iri("http://foaf.org/name")?),
    None,
))?;

// Complex pattern matching
let quoted_patterns = store.match_quoted_pattern(&StarPattern::new(
    None,
    Some(StarTerm::iri("http://example.org/confidence")?),
    None,
))?;

// Iterator-based matching (memory efficient)
for triple in store.iter_matching(&StarPattern::any())? {
    println!("Triple: {} {} {}", triple.subject(), triple.predicate(), triple.object());
}
```

### Graph Operations

```rust
use oxirs_star::{StarStore, StarGraph, StarTerm};

let mut store = StarStore::new();

// Named graph operations
let graph_name = StarTerm::iri("http://example.org/graph1")?;
let graph = StarGraph::new(graph_name.clone());

// Add triple to named graph
store.insert_in_graph(&triple, &graph_name)?;

// Query specific graph
let graph_triples = store.match_in_graph(
    &StarPattern::any(),
    &graph_name,
)?;

// List all graphs
let graphs = store.list_graphs()?;
println!("Available graphs: {:?}", graphs);

// Default graph operations
store.insert_in_default_graph(&triple)?;
let default_triples = store.match_in_default_graph(&StarPattern::any())?;
```

## Parser API

### Turtle-star Parser

```rust
use oxirs_star::parser::{TurtleStarParser, ParseOptions};

let turtle_star_data = r#"
@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

<<ex:alice foaf:knows ex:bob>> ex:confidence 0.9 .
<<ex:alice foaf:age 30>> ex:source ex:census2020 .
ex:alice foaf:name "Alice Smith" .
"#;

// Basic parsing
let parser = TurtleStarParser::new();
let triples = parser.parse(turtle_star_data)?;
println!("Parsed {} triples", triples.len());

// Parsing with options
let options = ParseOptions {
    base_iri: Some("http://example.org/base/".to_string()),
    strict_mode: false,
    max_nesting_depth: 10,
    validate_datatypes: true,
};

let parser_with_options = TurtleStarParser::with_options(options);
let triples = parser_with_options.parse(turtle_star_data)?;

// Streaming parser for large files
use std::fs::File;
let file = File::open("large_dataset.ttls")?;
let streaming_parser = TurtleStarParser::streaming();

for result in streaming_parser.parse_stream(file)? {
    match result {
        Ok(triple) => {
            // Process triple
            println!("Parsed: {}", triple);
        },
        Err(e) => {
            eprintln!("Parse error: {}", e);
            // Continue parsing or break based on error handling strategy
        }
    }
}

// Error recovery parsing
let malformed_data = r#"
@prefix ex: <http://example.org/> .
<<ex:alice foaf:knows ex:bob> ex:confidence 0.9 .  # Missing closing >>
ex:alice foaf:name "Alice Smith" .
"#;

let recovery_parser = TurtleStarParser::with_error_recovery();
let (triples, errors) = recovery_parser.parse_with_errors(malformed_data)?;
println!("Parsed {} triples with {} errors", triples.len(), errors.len());
```

### N-Triples-star Parser

```rust
use oxirs_star::parser::NTriplesStarParser;

let ntriples_star_data = r#"
<<<http://example.org/alice> <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob>>> <http://example.org/confidence> "0.9"^^<http://www.w3.org/2001/XMLSchema#decimal> .
<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice Smith" .
"#;

let parser = NTriplesStarParser::new();
let triples = parser.parse(ntriples_star_data)?;

// Parallel parsing for large files
let parallel_parser = NTriplesStarParser::parallel(4); // 4 threads
let triples = parallel_parser.parse_file("large_dataset.nts")?;
```

### TriG-star Parser

```rust
use oxirs_star::parser::TriGStarParser;

let trig_star_data = r#"
@prefix ex: <http://example.org/> .

ex:graph1 {
    <<ex:alice ex:knows ex:bob>> ex:confidence 0.9 .
    ex:alice ex:name "Alice" .
}

ex:graph2 {
    <<ex:bob ex:knows ex:charlie>> ex:confidence 0.8 .
    ex:bob ex:name "Bob" .
}
"#;

let parser = TriGStarParser::new();
let dataset = parser.parse(trig_star_data)?;

// Access individual graphs
for (graph_name, triples) in dataset.iter() {
    println!("Graph {}: {} triples", graph_name, triples.len());
}
```

### N-Quads-star Parser

```rust
use oxirs_star::parser::NQuadsStarParser;

let nquads_star_data = r#"
<<<http://example.org/alice> <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob>>> <http://example.org/confidence> "0.9"^^<http://www.w3.org/2001/XMLSchema#decimal> <http://example.org/graph1> .
<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice Smith" <http://example.org/graph1> .
"#;

let parser = NQuadsStarParser::new();
let quads = parser.parse(nquads_star_data)?;

for quad in quads {
    println!("Quad: {} {} {} in graph {}", 
        quad.subject(), quad.predicate(), quad.object(), 
        quad.graph().map(|g| g.to_string()).unwrap_or("default".to_string())
    );
}
```

## Serializer API

### Turtle-star Serializer

```rust
use oxirs_star::serializer::{TurtleStarSerializer, SerializerOptions};

let mut store = StarStore::new();
// ... populate store ...

// Basic serialization
let serializer = TurtleStarSerializer::new();
let turtle_star_output = serializer.serialize(&store)?;
println!("{}", turtle_star_output);

// Serialization with options
let options = SerializerOptions {
    pretty_print: true,
    base_iri: Some("http://example.org/".to_string()),
    use_prefixes: true,
    max_line_length: 80,
    sort_predicates: true,
};

let serializer = TurtleStarSerializer::with_options(options);
let formatted_output = serializer.serialize(&store)?;

// Streaming serialization for large datasets
use std::fs::File;
let output_file = File::create("output.ttls")?;
let streaming_serializer = TurtleStarSerializer::streaming(output_file);
streaming_serializer.serialize_stream(store.iter())?;

// Custom prefix management
let mut serializer = TurtleStarSerializer::new();
serializer.add_prefix("ex", "http://example.org/")?;
serializer.add_prefix("foaf", "http://xmlns.com/foaf/0.1/")?;
let prefixed_output = serializer.serialize(&store)?;
```

### N-Triples-star Serializer

```rust
use oxirs_star::serializer::NTriplesStarSerializer;

let serializer = NTriplesStarSerializer::new();
let ntriples_output = serializer.serialize(&store)?;

// Parallel serialization
let parallel_serializer = NTriplesStarSerializer::parallel(4);
let output = parallel_serializer.serialize_large(&store)?;
```

### JSON-LD-star Serializer (Future)

```rust
// Coming soon in future releases
use oxirs_star::serializer::JsonLdStarSerializer;

let serializer = JsonLdStarSerializer::new();
let json_ld_output = serializer.serialize(&store)?;
```

## Query Engine API

### Basic SPARQL-star Queries

```rust
use oxirs_star::query::{StarQueryEngine, QueryOptions};

let engine = StarQueryEngine::new(&store);

// Simple SPARQL-star query
let query = r#"
PREFIX ex: <http://example.org/>
SELECT ?stmt ?conf WHERE {
    ?stmt ex:confidence ?conf .
    ?stmt { ?s ex:knows ?o }
    FILTER(?conf > 0.8)
}
ORDER BY DESC(?conf)
"#;

let results = engine.execute(query)?;
for binding in results {
    println!("Statement: {}, Confidence: {}", 
        binding.get("stmt").unwrap(), 
        binding.get("conf").unwrap()
    );
}

// Query with options
let options = QueryOptions {
    timeout: Some(std::time::Duration::from_secs(30)),
    max_results: Some(1000),
    enable_optimization: true,
    explain_plan: false,
};

let results = engine.execute_with_options(query, options)?;

// CONSTRUCT queries with quoted triples
let construct_query = r#"
PREFIX ex: <http://example.org/>
CONSTRUCT {
    <<?s ex:knows ?o>> ex:inferredConfidence ?newConf
} WHERE {
    ?s ex:knows ?o .
    ?s ex:trustScore ?trust .
    ?o ex:reliabilityScore ?rel .
    BIND((?trust + ?rel) / 2 AS ?newConf)
}
"#;

let constructed_triples = engine.construct(construct_query)?;
```

### Advanced Query Patterns

```rust
// Nested quoted triple queries
let nested_query = r#"
PREFIX ex: <http://example.org/>
SELECT ?meta ?value WHERE {
    ?meta ex:describes <<?person ex:believes <<?subj ex:prop ?obj>>>> .
    ?meta ex:value ?value .
}
"#;

// Property path queries with quoted triples
let path_query = r#"
PREFIX ex: <http://example.org/>
SELECT ?start ?end WHERE {
    <<?start ex:knows+ ?end>> ex:pathConfidence ?conf .
    FILTER(?conf > 0.7)
}
"#;

// Aggregation with quoted triples
let aggregation_query = r#"
PREFIX ex: <http://example.org/>
SELECT ?person (AVG(?conf) AS ?avgConf) WHERE {
    <<?person ?prop ?value>> ex:confidence ?conf .
    GROUP BY ?person
    HAVING(?avgConf > 0.8)
}
ORDER BY DESC(?avgConf)
"#;

let results = engine.execute(aggregation_query)?;
```

### Federated SPARQL-star Queries

```rust
use oxirs_star::query::federation::FederatedQueryEngine;

let federated_engine = FederatedQueryEngine::new();
federated_engine.add_endpoint("local", &store)?;
federated_engine.add_remote_endpoint(
    "remote", 
    "https://example.org/sparql-star",
    Some(std::time::Duration::from_secs(10))
)?;

let federated_query = r#"
PREFIX ex: <http://example.org/>
SELECT ?stmt ?localConf ?remoteConf WHERE {
    SERVICE <local> {
        ?stmt ex:confidence ?localConf .
    }
    SERVICE <remote> {
        ?stmt ex:confidence ?remoteConf .
    }
    FILTER(?localConf != ?remoteConf)
}
"#;

let results = federated_engine.execute(federated_query).await?;
```

## SPARQL-star Functions

### Built-in Functions

```rust
// Available SPARQL-star functions in queries:

// TRIPLE() - Create quoted triple
let triple_query = r#"
SELECT ?qt WHERE {
    ?s ?p ?o .
    BIND(TRIPLE(?s, ?p, ?o) AS ?qt)
}
"#;

// SUBJECT(), PREDICATE(), OBJECT() - Extract from quoted triple
let extract_query = r#"
SELECT ?s ?p ?o WHERE {
    ?qt ex:confidence ?conf .
    BIND(SUBJECT(?qt) AS ?s)
    BIND(PREDICATE(?qt) AS ?p)
    BIND(OBJECT(?qt) AS ?o)
    FILTER(?conf > 0.8)
}
"#;

// isTRIPLE() - Check if term is quoted triple
let check_query = r#"
SELECT ?term WHERE {
    ?term ?pred ?obj .
    FILTER(isTRIPLE(?term))
}
"#;
```

### Custom Functions

```rust
use oxirs_star::query::functions::{FunctionRegistry, CustomFunction};

let mut registry = FunctionRegistry::new();

// Register custom similarity function
let similarity_fn = CustomFunction::new(
    "http://example.org/similarity",
    |args| {
        // Implementation for custom similarity calculation
        // This would integrate with vector search capabilities
        Ok(StarTerm::typed_literal("0.85", "http://www.w3.org/2001/XMLSchema#decimal")?)
    }
);

registry.register("similarity", similarity_fn)?;

// Use in queries
let custom_query = r#"
PREFIX ex: <http://example.org/>
PREFIX fn: <http://example.org/>
SELECT ?entity1 ?entity2 ?sim WHERE {
    ?entity1 a ex:Person .
    ?entity2 a ex:Person .
    BIND(fn:similarity(?entity1, ?entity2) AS ?sim)
    FILTER(?sim > 0.8)
}
"#;
```

## Reification API

### Automatic Reification

```rust
use oxirs_star::reification::{ReificationStrategy, ReificationEngine};

let reification_engine = ReificationEngine::new(ReificationStrategy::Standard);

// Convert RDF-star to reified RDF
let rdf_star_triple = StarTriple::new(
    StarTerm::quoted_triple(StarTriple::new(
        StarTerm::iri("http://example.org/alice")?,
        StarTerm::iri("http://foaf.org/knows")?,
        StarTerm::iri("http://example.org/bob")?,
    )?),
    StarTerm::iri("http://example.org/confidence")?,
    StarTerm::literal("0.9"),
)?;

let reified_triples = reification_engine.reify(&rdf_star_triple)?;
println!("Reified into {} triples", reified_triples.len());

// Convert back from reified RDF to RDF-star
let star_triples = reification_engine.dereify(&reified_triples)?;
```

### Custom Reification Strategies

```rust
use oxirs_star::reification::{CustomReificationStrategy, ReificationContext};

let custom_strategy = CustomReificationStrategy::new()
    .with_statement_class("http://example.org/Statement")
    .with_subject_property("http://example.org/hasSubject")
    .with_predicate_property("http://example.org/hasPredicate")
    .with_object_property("http://example.org/hasObject");

let engine = ReificationEngine::new(ReificationStrategy::Custom(custom_strategy));

// Apply custom reification
let reified = engine.reify_with_context(&triple, &ReificationContext::new())?;
```

### Hybrid Reification

```rust
use oxirs_star::reification::HybridReificationEngine;

// Supports both RDF-star and reified RDF simultaneously
let hybrid_engine = HybridReificationEngine::new();

// Query can use both forms
let hybrid_query = r#"
PREFIX ex: <http://example.org/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT ?confidence WHERE {
    {
        # RDF-star form
        <<?s ex:knows ?o>> ex:confidence ?confidence .
    } UNION {
        # Reified form  
        ?stmt rdf:subject ?s ;
              rdf:predicate ex:knows ;
              rdf:object ?o ;
              ex:confidence ?confidence .
    }
}
"#;

let results = hybrid_engine.execute(hybrid_query)?;
```

## Configuration API

### Engine Configuration

```rust
use oxirs_star::config::{StarConfig, ParsingConfig, QueryConfig, StorageConfig};

let config = StarConfig::builder()
    .parsing(ParsingConfig {
        max_nesting_depth: 20,
        strict_mode: false,
        enable_error_recovery: true,
        parallel_parsing: true,
        buffer_size: 64 * 1024,
    })
    .query(QueryConfig {
        enable_optimization: true,
        timeout: std::time::Duration::from_secs(60),
        max_results: 10_000,
        enable_federation: true,
        cache_size: 1000,
    })
    .storage(StorageConfig {
        enable_indexing: true,
        index_quoted_triples: true,
        memory_limit: 2 * 1024 * 1024 * 1024, // 2GB
        disk_cache_size: 1024 * 1024 * 1024, // 1GB
        enable_compression: true,
    })
    .build();

// Apply configuration globally
oxirs_star::init_with_config(config)?;

// Or create engine with specific config
let engine = StarQueryEngine::with_config(&store, config)?;
```

### Runtime Configuration Updates

```rust
use oxirs_star::config::RuntimeConfig;

let mut runtime_config = RuntimeConfig::current();

// Update configuration at runtime
runtime_config.update_memory_limit(4 * 1024 * 1024 * 1024)?; // 4GB
runtime_config.update_cache_size(2000)?;
runtime_config.enable_parallel_processing(8)?; // 8 threads

// Configuration monitoring
let stats = runtime_config.get_performance_stats()?;
println!("Memory usage: {} MB", stats.memory_usage / 1024 / 1024);
println!("Cache hit rate: {:.2}%", stats.cache_hit_rate * 100.0);
```

## Error Handling

### Error Types

```rust
use oxirs_star::{StarError, StarErrorKind, StarResult};

// Comprehensive error handling
fn process_rdf_star_data(data: &str) -> StarResult<Vec<StarTriple>> {
    match StarStore::parse_turtle_star(data) {
        Ok(triples) => Ok(triples),
        Err(StarError { kind: StarErrorKind::ParseError, message, .. }) => {
            eprintln!("Parsing failed: {}", message);
            // Attempt recovery
            match StarStore::parse_turtle_star_with_recovery(data) {
                Ok((triples, errors)) => {
                    eprintln!("Recovered {} triples with {} errors", triples.len(), errors.len());
                    Ok(triples)
                },
                Err(e) => Err(e)
            }
        },
        Err(StarError { kind: StarErrorKind::ValidationError, .. }) => {
            eprintln!("Validation failed, attempting with relaxed validation");
            StarStore::parse_with_options(data, ParseOptions::relaxed())
        },
        Err(e) => Err(e)
    }
}

// Error chaining and context
use anyhow::{Context, Result};

fn load_and_process_file(path: &str) -> Result<StarStore> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read file: {}", path))?;
    
    let triples = process_rdf_star_data(&content)
        .with_context(|| "Failed to parse RDF-star data")?;
    
    let mut store = StarStore::new();
    store.insert_batch(&triples)
        .with_context(|| "Failed to insert triples into store")?;
    
    Ok(store)
}
```

### Error Recovery Strategies

```rust
use oxirs_star::error::{ErrorRecoveryStrategy, RecoveryOptions};

let recovery_options = RecoveryOptions {
    continue_on_parse_error: true,
    max_errors: 100,
    skip_malformed_triples: true,
    report_recovered_errors: true,
};

let recovery_strategy = ErrorRecoveryStrategy::Lenient(recovery_options);

let parser = TurtleStarParser::with_recovery(recovery_strategy);
let (triples, errors) = parser.parse_with_error_collection(data)?;

// Process errors for reporting
for error in errors {
    match error.kind {
        ErrorKind::MalformedQuotedTriple => {
            log::warn!("Skipped malformed quoted triple at line {}", error.line);
        },
        ErrorKind::InvalidIRI => {
            log::error!("Invalid IRI at line {}: {}", error.line, error.context);
        },
        _ => {
            log::debug!("Minor parsing issue: {}", error.message);
        }
    }
}
```

## Advanced Patterns

### Streaming Processing

```rust
use oxirs_star::streaming::{StreamProcessor, StreamingOptions};
use tokio_stream::StreamExt;

let options = StreamingOptions {
    buffer_size: 1024 * 1024, // 1MB buffer
    parallel_processing: true,
    thread_count: 4,
    backpressure_threshold: 10000,
};

let mut processor = StreamProcessor::with_options(options);

// Process large RDF-star files
let file_stream = tokio::fs::File::open("huge_dataset.ttls").await?;
let mut triple_stream = processor.parse_stream(file_stream);

while let Some(result) = triple_stream.next().await {
    match result {
        Ok(triple) => {
            // Process triple in real-time
            process_triple_async(triple).await?;
        },
        Err(e) => {
            log::error!("Streaming error: {}", e);
        }
    }
}

// Streaming validation
let validator = StreamingValidator::new(shapes);
let mut validation_stream = validator.validate_stream(triple_stream);

while let Some(validation_result) = validation_stream.next().await {
    match validation_result {
        Ok(report) if report.conforms() => {
            // Triple is valid
            validated_count += 1;
        },
        Ok(report) => {
            // Handle violations
            for violation in report.violations() {
                log::warn!("Validation violation: {}", violation.message());
            }
        },
        Err(e) => {
            log::error!("Validation error: {}", e);
        }
    }
}
```

### Concurrent Access Patterns

```rust
use oxirs_star::concurrent::{ConcurrentStarStore, ReadWriteLock};
use std::sync::Arc;
use tokio::task;

let store = Arc::new(ConcurrentStarStore::new());

// Concurrent readers
let reader_tasks: Vec<_> = (0..10).map(|i| {
    let store = store.clone();
    task::spawn(async move {
        let pattern = StarPattern::new(
            Some(StarTerm::iri(&format!("http://example.org/reader{}", i))?),
            None,
            None,
        );
        let results = store.match_pattern(&pattern).await?;
        println!("Reader {}: found {} triples", i, results.len());
        Ok::<_, StarError>(results)
    })
}).collect();

// Concurrent writer
let writer_store = store.clone();
let writer_task = task::spawn(async move {
    for i in 0..100 {
        let triple = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/writer/item{}", i))?,
            StarTerm::iri("http://example.org/hasIndex")?,
            StarTerm::typed_literal(&i.to_string(), "http://www.w3.org/2001/XMLSchema#integer")?,
        )?;
        writer_store.insert(&triple).await?;
    }
    Ok::<_, StarError>(())
});

// Wait for all tasks
for task in reader_tasks {
    task.await??;
}
writer_task.await??;
```

### Memory-Mapped Storage

```rust
use oxirs_star::storage::{MemoryMappedStore, StorageOptions};

let storage_options = StorageOptions {
    file_path: "/path/to/rdf_star_data.db".to_string(),
    initial_size: 1024 * 1024 * 1024, // 1GB
    grow_increment: 256 * 1024 * 1024, // 256MB
    enable_compression: true,
    sync_on_write: false,
};

let mmap_store = MemoryMappedStore::create_or_open(storage_options)?;

// Large-scale operations are now disk-backed
for i in 0..1_000_000 {
    let triple = create_sample_triple(i)?;
    mmap_store.insert(&triple)?;
}

// Efficient iteration over large datasets
for triple in mmap_store.iter() {
    // Process without loading everything into memory
    process_triple_efficiently(&triple)?;
}
```

## Performance Optimization

### Indexing Strategies

```rust
use oxirs_star::indexing::{IndexManager, IndexType, IndexConfig};

let mut index_manager = IndexManager::new();

// Create specialized indices
index_manager.create_index(
    "subject_index",
    IndexType::BTree,
    IndexConfig {
        index_subjects: true,
        index_predicates: false,
        index_objects: false,
        index_quoted_triples: true,
    }
)?;

index_manager.create_index(
    "full_text_index",
    IndexType::FullText,
    IndexConfig {
        index_literals: true,
        enable_stemming: true,
        enable_phonetic: false,
        languages: vec!["en".to_string(), "fr".to_string()],
    }
)?;

// Query with index hints
let query_with_hints = r#"
PREFIX ex: <http://example.org/>
SELECT ?s ?o WHERE {
    ?s ex:hasText ?text .
    ?s ex:relatedTo ?o .
    FILTER(contains(?text, "machine learning"))
} 
OPTION (INDEX "full_text_index")
"#;

let optimized_results = engine.execute_with_hints(query_with_hints)?;
```

### Caching Strategies

```rust
use oxirs_star::cache::{QueryCache, ResultCache, LRUCache};

// Multi-level caching
let query_cache = QueryCache::new(1000); // Cache 1000 query plans
let result_cache = ResultCache::new(
    LRUCache::with_capacity(500), // Cache 500 result sets
    std::time::Duration::from_secs(300), // 5 minute TTL
);

let cached_engine = StarQueryEngine::builder()
    .with_store(&store)
    .with_query_cache(query_cache)
    .with_result_cache(result_cache)
    .build();

// Warm up cache with common queries
let common_queries = load_common_queries()?;
for query in common_queries {
    cached_engine.prepare_query(&query)?; // Compiles and caches plan
}

// Subsequent executions will be faster
let results = cached_engine.execute(query)?; // Uses cached plan and potentially cached results
```

### Batch Processing Optimization

```rust
use oxirs_star::batch::{BatchProcessor, BatchConfig};

let batch_config = BatchConfig {
    batch_size: 10000,
    parallel_batches: 4,
    memory_limit: 512 * 1024 * 1024, // 512MB per batch
    enable_compression: true,
};

let batch_processor = BatchProcessor::with_config(batch_config);

// Process large datasets efficiently
let large_dataset = load_large_dataset()?;
let results = batch_processor.process_in_batches(
    large_dataset,
    |batch| {
        // Process each batch
        let mut local_store = StarStore::new();
        local_store.insert_batch(&batch)?;
        
        // Run queries on batch
        let query_results = engine.execute_on_store(query, &local_store)?;
        Ok(query_results)
    }
)?;

// Merge results from all batches
let final_results = batch_processor.merge_results(results)?;
```

### Monitoring and Profiling

```rust
use oxirs_star::profiling::{Profiler, ProfilerConfig, MetricCollector};

let profiler_config = ProfilerConfig {
    collect_query_stats: true,
    collect_memory_stats: true,
    collect_index_stats: true,
    sample_rate: 0.1, // Sample 10% of operations
};

let profiler = Profiler::with_config(profiler_config);
let engine = StarQueryEngine::with_profiler(&store, profiler);

// Execute queries with profiling
let results = engine.execute(query)?;

// Get performance metrics
let metrics = engine.get_profiling_metrics()?;
println!("Query execution time: {:?}", metrics.query_time);
println!("Memory peak usage: {} MB", metrics.peak_memory / 1024 / 1024);
println!("Index hits: {}, misses: {}", metrics.index_hits, metrics.index_misses);

// Export metrics for monitoring systems
let prometheus_metrics = metrics.to_prometheus_format()?;
let json_metrics = serde_json::to_string(&metrics)?;
```

This comprehensive API reference covers all major aspects of the OxiRS-Star RDF-star implementation, providing production-ready examples and advanced usage patterns for building robust semantic web applications.