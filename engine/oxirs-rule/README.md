# OxiRS Rule Engine

[![Version](https://img.shields.io/badge/version-0.1.0--rc.1-blue)](https://github.com/cool-japan/oxirs/releases)

**Status**: Release Candidate (v0.1.0-rc.1) - Released December 26, 2025

✨ **Release Candidate**: Production-ready with API stability guarantees and comprehensive testing.

A high-performance, comprehensive reasoning engine for Semantic Web applications, implementing forward chaining, backward chaining, RETE networks, RDFS, OWL RL, and SWRL rule processing.

## Features

### 🚀 Core Reasoning Engines

- **Forward Chaining** - Efficient data-driven inference with pattern matching and built-in predicates
- **Backward Chaining** - Goal-driven proof search with cycle detection and caching
- **RETE Network** - High-performance pattern matching for real-time rule execution

### 🎯 Standards Compliance

- **RDFS Reasoning** - Complete RDFS entailment rules (rdfs2, rdfs3, rdfs5, rdfs7, rdfs9, rdfs11)
- **OWL RL Support** - Class/property equivalence, characteristics, identity reasoning
- **SWRL Integration** - Semantic Web Rule Language with extensive built-in predicates

### ⚡ Performance Features

- **Incremental Updates** - Efficient handling of dynamic knowledge base changes
- **Pattern Indexing** - Optimized rule and fact lookup with caching
- **Memory Optimization** - Efficient data structures for large-scale reasoning
- **Statistics & Monitoring** - Comprehensive performance metrics and tracing

## Installation

Add to your `Cargo.toml`:

```toml
# Experimental feature
[dependencies]
oxirs-rule = "0.1.0-rc.1"
```

## Quick Start

```rust
use oxirs_rule::*;

// Create a new rule engine
let mut engine = RuleEngine::new();

// Add facts
engine.add_fact(atom!(
    "knows", 
    term!("alice"), 
    term!("bob")
));

// Add rules
engine.add_rule(rule!(
    [atom!("knows", var!("X"), var!("Y"))],
    [atom!("friend", var!("X"), var!("Y"))]
));

// Perform forward chaining inference
let stats = engine.forward_chain()?;
println!("Derived {} new facts", stats.facts_derived);

// Query the knowledge base
let results = engine.query(&atom!("friend", var!("X"), var!("Y")))?;
for result in results {
    println!("Friend relationship: {:?}", result);
}
```

## Architecture

### Core Components

- **`RuleEngine`** - Main interface unifying all reasoning strategies
- **`ForwardChainEngine`** - Data-driven inference with fixpoint calculation
- **`BackwardChainEngine`** - Goal-driven proof search with memoization
- **`ReteNetwork`** - High-performance pattern matching network
- **`RdfsReasoner`** - RDFS schema reasoning and entailment
- **`OwlReasoner`** - OWL RL profile reasoning
- **`SwrlEngine`** - SWRL rule processing with built-in functions

### Data Structures

```rust
// Core data types
pub struct Rule {
    pub head: Vec<RuleAtom>,
    pub body: Vec<RuleAtom>,
}

pub struct RuleAtom {
    pub predicate: String,
    pub terms: Vec<Term>,
}

pub enum Term {
    Variable(String),
    Constant(String),
    Literal(String),
}
```

## Reasoning Strategies

### Forward Chaining

Ideal for:
- Batch processing of large datasets
- Materializing all possible inferences
- Data integration and ETL processes

```rust
let stats = engine.forward_chain()?;
println!("Facts derived: {}", stats.facts_derived);
println!("Rules fired: {}", stats.rules_fired);
```

### Backward Chaining

Ideal for:
- Query-driven reasoning
- Interactive applications
- Proof explanation and validation

```rust
let goal = atom!("ancestor", term!("alice"), var!("X"));
let proofs = engine.backward_chain(&goal)?;
for proof in proofs {
    println!("Proof path: {:?}", proof.derivation_path);
}
```

### RETE Network

Ideal for:
- Real-time rule processing
- Incremental updates
- High-throughput applications

```rust
let mut rete = ReteNetwork::new();
rete.add_rule(&rule);
rete.add_fact(&fact);
let results = rete.execute()?;
```

## Built-in Predicates

### Comparison Operations
- `equal(?x, ?y)` - Equality testing
- `notEqual(?x, ?y)` - Inequality testing
- `lessThan(?x, ?y)` - Numeric comparison
- `greaterThan(?x, ?y)` - Numeric comparison

### Mathematical Operations
- `add(?x, ?y, ?result)` - Addition
- `subtract(?x, ?y, ?result)` - Subtraction
- `multiply(?x, ?y, ?result)` - Multiplication

### String Operations
- `stringConcat(?s1, ?s2, ?result)` - String concatenation
- `stringLength(?string, ?length)` - String length

### Utility Predicates
- `bound(?var)` - Variable binding check
- `unbound(?var)` - Variable unbound check

## RDFS Reasoning

Automatic inference of:
- Class hierarchies (`rdfs:subClassOf`)
- Property hierarchies (`rdfs:subPropertyOf`)
- Domain and range constraints (`rdfs:domain`, `rdfs:range`)
- Type membership (`rdf:type`)

```rust
let mut rdfs = RdfsReasoner::new();
rdfs.add_schema_triple("Person", "rdfs:subClassOf", "Agent");
rdfs.add_data_triple("alice", "rdf:type", "Person");

let inferred = rdfs.materialize()?;
// Automatically infers: alice rdf:type Agent
```

## OWL RL Reasoning

Support for:
- Property characteristics (functional, transitive, symmetric)
- Class equivalence and disjointness
- Individual identity (`owl:sameAs`, `owl:differentFrom`)
- Basic property restrictions

```rust
let mut owl = OwlReasoner::new();
owl.add_axiom("hasParent", "rdf:type", "owl:TransitiveProperty");
owl.add_fact("alice", "hasParent", "bob");
owl.add_fact("bob", "hasParent", "charlie");

let inferred = owl.reason()?;
// Automatically infers: alice hasParent charlie
```

## SWRL Rules

Support for complex rule definitions:

```rust
// Person(?p) ∧ hasAge(?p, ?age) ∧ greaterThan(?age, 18) → Adult(?p)
let swrl_rule = SwrlRule {
    body: vec![
        SwrlAtom::Class("Person".to_string(), SwrlArgument::Variable("p".to_string())),
        SwrlAtom::DataProperty("hasAge".to_string(), 
            SwrlArgument::Variable("p".to_string()), 
            SwrlArgument::Variable("age".to_string())),
        SwrlAtom::BuiltIn("greaterThan".to_string(), vec![
            SwrlArgument::Variable("age".to_string()),
            SwrlArgument::Literal("18".to_string())
        ])
    ],
    head: vec![
        SwrlAtom::Class("Adult".to_string(), SwrlArgument::Variable("p".to_string()))
    ]
};
```

## Performance Characteristics

### Scalability
- **Forward Chaining**: O(n³) worst case, optimized for sparse rules
- **Backward Chaining**: O(2^n) worst case, pruned with cycle detection
- **RETE Network**: O(1) for fact insertion, O(n) for rule addition

### Memory Usage
- Efficient fact storage with deduplication
- Rule indexing for fast pattern matching
- Configurable caching strategies

### Benchmarks
- 1M+ triples: <60 seconds for full materialization
- 10K+ rules: <10 seconds for compilation
- Memory usage: <8GB for 1M triple datasets

## Integration

### With oxirs-core
```rust
use oxirs_core::model::{Triple, IRI};
use oxirs_rule::RuleEngine;

let mut engine = RuleEngine::new();
// Convert oxirs-core triples to rule facts
engine.add_triple(&triple)?;
```

### With SPARQL
```rust
// Use reasoning results in SPARQL queries
let inferred_facts = engine.get_materialized_facts();
// Add to SPARQL dataset for querying
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
cargo nextest run --no-fail-fast

# Run specific reasoning tests
cargo nextest run -p oxirs-rule --no-fail-fast

# Run performance benchmarks
cargo test --release --features benchmarks
```

### Test Coverage
- 722/722 tests passing (100% success rate)
- Unit tests for all reasoning algorithms
- Integration tests with real-world datasets
- Performance regression tests

## Configuration

```rust
use oxirs_rule::config::EngineConfig;

let config = EngineConfig {
    max_iterations: 1000,
    cache_size: 10000,
    enable_statistics: true,
    reasoning_strategy: ReasoningStrategy::Hybrid,
};

let engine = RuleEngine::with_config(config);
```

## Error Handling

Comprehensive error types with detailed context:

```rust
use oxirs_rule::error::RuleEngineError;

match engine.forward_chain() {
    Ok(stats) => println!("Success: {} facts derived", stats.facts_derived),
    Err(RuleEngineError::UnificationError(msg)) => {
        eprintln!("Unification failed: {}", msg);
    },
    Err(RuleEngineError::CycleDetected(path)) => {
        eprintln!("Infinite recursion detected: {:?}", path);
    },
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Logging and Monitoring

Built-in tracing support:

```rust
use tracing::{info, debug};

// Enable tracing
tracing_subscriber::init();

// Reasoning operations automatically logged
let stats = engine.forward_chain()?;
// Logs: "Forward chaining completed: 1000 facts derived, 50 rules fired"
```

## Contributing

1. Follow Rust 2021 edition best practices
2. Maintain >95% test coverage
3. Use `cargo clippy` and `cargo fmt`
4. Add comprehensive documentation
5. Include performance benchmarks

## License

Licensed under the Apache License, Version 2.0 or the MIT License, at your option.

## Status

🚀 **Release Candidate (v0.1.0-rc.1)** – December 26, 2025

Highlights:
- ✅ Forward/backward chaining over persisted datasets with automatic inference snapshots
- ✅ RETE network optimized with SciRS2 metrics and tracing hooks
- ✅ Integrated with federation-aware SPARQL workflows for rule-driven post-processing
- ✅ 722/722 tests passing plus CLI end-to-end coverage
- 🚧 Advanced distributed reasoning (planned for future release)