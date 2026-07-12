# OxiRS Rule Engine

[![Version](https://img.shields.io/badge/version-0.3.2-blue)](https://github.com/cool-japan/oxirs/releases)

**Status**: v0.3.2 - Released 2026-07-12

✨ **Production Release**: Production-ready with API stability guarantees and comprehensive testing.

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
oxirs-rule = "0.3.2"
```

## Quick Start

```rust
use oxirs_rule::{RuleEngine, Rule, RuleAtom, Term};

# fn example() -> anyhow::Result<()> {
// Create a new rule engine
let mut engine = RuleEngine::new();

// Add a rule: knows(X,Y) -> friend(X,Y)
engine.add_rule(Rule {
    name: "friend_rule".to_string(),
    body: vec![RuleAtom::Triple {
        subject: Term::Variable("X".to_string()),
        predicate: Term::Constant("knows".to_string()),
        object: Term::Variable("Y".to_string()),
    }],
    head: vec![RuleAtom::Triple {
        subject: Term::Variable("X".to_string()),
        predicate: Term::Constant("friend".to_string()),
        object: Term::Variable("Y".to_string()),
    }],
});

// Facts to reason over
let facts = vec![RuleAtom::Triple {
    subject: Term::Constant("alice".to_string()),
    predicate: Term::Constant("knows".to_string()),
    object: Term::Constant("bob".to_string()),
}];

// Perform forward chaining inference
let derived = engine.forward_chain(&facts)?;
println!("Derived {} facts", derived.len());
for fact in &derived {
    println!("Derived fact: {:?}", fact);
}
# Ok(())
# }
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
    pub name: String,
    pub body: Vec<RuleAtom>,
    pub head: Vec<RuleAtom>,
}

pub enum RuleAtom {
    Triple { subject: Term, predicate: Term, object: Term },
    Builtin { name: String, args: Vec<Term> },
    NotEqual { left: Term, right: Term },
    GreaterThan { left: Term, right: Term },
    LessThan { left: Term, right: Term },
}

pub enum Term {
    Variable(String),
    Constant(String),
    Literal(String),
    Function { name: String, args: Vec<Term> },
}
```

## Reasoning Strategies

### Forward Chaining

Ideal for:
- Batch processing of large datasets
- Materializing all possible inferences
- Data integration and ETL processes

```rust
let derived = engine.forward_chain(&facts)?;
println!("Facts derived: {}", derived.len());
```

### Backward Chaining

Ideal for:
- Query-driven reasoning
- Interactive applications
- Proof explanation and validation

```rust
let goal = RuleAtom::Triple {
    subject: Term::Constant("alice".to_string()),
    predicate: Term::Constant("ancestor".to_string()),
    object: Term::Variable("X".to_string()),
};
let provable = engine.backward_chain(&goal)?;
println!("Goal provable: {}", provable);
```

### RETE Network

Ideal for:
- Real-time rule processing
- Incremental updates
- High-throughput applications

```rust
use oxirs_rule::rete::ReteNetwork;

let mut rete = ReteNetwork::new();
rete.add_rule(&rule)?;
let results = rete.forward_chain(vec![fact])?;
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
use oxirs_rule::rdfs::RdfsReasoner;
use oxirs_rule::{RuleAtom, Term};

let mut rdfs = RdfsReasoner::new();

let schema_and_data = vec![
    RuleAtom::Triple {
        subject: Term::Constant("Person".to_string()),
        predicate: Term::Constant("rdfs:subClassOf".to_string()),
        object: Term::Constant("Agent".to_string()),
    },
    RuleAtom::Triple {
        subject: Term::Constant("alice".to_string()),
        predicate: Term::Constant("rdf:type".to_string()),
        object: Term::Constant("Person".to_string()),
    },
];

let inferred = rdfs.infer(&schema_and_data)?;
// `inferred` includes the entailed triple: alice rdf:type Agent
```

## OWL RL Reasoning

Support for:
- Property characteristics (functional, transitive, symmetric)
- Class equivalence and disjointness
- Individual identity (`owl:sameAs`, `owl:differentFrom`)
- Basic property restrictions

```rust
use oxirs_rule::owl::{vocabulary, OwlReasoner};
use oxirs_rule::{RuleAtom, Term};

let mut owl = OwlReasoner::new();
owl.context
    .set_property_characteristic("hasParent", vocabulary::OWL_TRANSITIVE_PROPERTY, true);

let facts = vec![
    RuleAtom::Triple {
        subject: Term::Constant("alice".to_string()),
        predicate: Term::Constant("hasParent".to_string()),
        object: Term::Constant("bob".to_string()),
    },
    RuleAtom::Triple {
        subject: Term::Constant("bob".to_string()),
        predicate: Term::Constant("hasParent".to_string()),
        object: Term::Constant("charlie".to_string()),
    },
];

let inferred = owl.infer(&facts)?;
// `inferred` includes the entailed triple: alice hasParent charlie
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
use oxirs_rule::{RuleEngine, RuleAtom, Term};

let mut engine = RuleEngine::new();
// Convert an oxirs-core triple's subject/predicate/object strings into a rule fact
engine.add_fact(RuleAtom::Triple {
    subject: Term::Constant(subject_iri),
    predicate: Term::Constant(predicate_iri),
    object: Term::Constant(object_iri),
});
```

### With SPARQL
```rust
// Use reasoning results in SPARQL queries
let inferred_facts = engine.get_facts();
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
- 2,240 tests passing (100% success rate)
- Unit tests for all reasoning algorithms
- Integration tests with real-world datasets
- Performance regression tests

## Configuration

```rust
use oxirs_rule::RuleEngine;

let mut engine = RuleEngine::new();

// Caching is enabled by default; toggle explicitly if needed
engine.enable_cache();

// Limit backward-chaining recursion depth (default: 100)
engine.set_backward_chain_max_depth(30);

if let Some(stats) = engine.get_cache_statistics() {
    println!("Cache statistics: {:?}", stats);
}
```

## Error Handling

Comprehensive error types with detailed context:

```rust
// `RuleEngine` reasoning methods return `anyhow::Result<T>`
match engine.forward_chain(&facts) {
    Ok(derived) => println!("Success: {} facts derived", derived.len()),
    Err(e) => eprintln!("Forward chaining failed: {e}"),
}
```

## Logging and Monitoring

Built-in tracing support:

```rust
use tracing::{info, debug};

// Enable tracing
tracing_subscriber::init();

// Reasoning operations automatically logged
let derived = engine.forward_chain(&facts)?;
// Logs (via tracing): "Forward chaining completed: N facts derived"
```

## Contributing

1. Follow Rust 2021 edition best practices
2. Maintain >95% test coverage
3. Use `cargo clippy` and `cargo fmt`
4. Add comprehensive documentation
5. Include performance benchmarks

## License

Licensed under the Apache License, Version 2.0.

## Status

🚀 **Production Release (v0.3.2)** – 2026-07-12

Highlights:
- ✅ Forward/backward chaining over persisted datasets with automatic inference snapshots
- ✅ RETE network optimized with SciRS2 metrics and tracing hooks
- ✅ Integrated with federation-aware SPARQL workflows for rule-driven post-processing
- ✅ 2,240 tests passing plus CLI end-to-end coverage
- 🚧 Advanced distributed reasoning (planned for future release)