# OxiRS SHACL 🔍

[![Crates.io](https://img.shields.io/crates/v/oxirs-shacl.svg)](https://crates.io/crates/oxirs-shacl)
[![Documentation](https://docs.rs/oxirs-shacl/badge.svg)](https://docs.rs/oxirs-shacl)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Build Status](https://github.com/cool-japan/oxirs/workflows/CI/badge.svg)](https://github.com/cool-japan/oxirs/actions)

A high-performance, W3C-compliant SHACL (Shapes Constraint Language) validator for RDF data, implemented in Rust as part of the OxiRS ecosystem.

## 🎯 Features

### Core SHACL Support
- **Complete SHACL Core Implementation** - All W3C SHACL constraint components
- **SHACL-SPARQL Extensions** - Advanced SPARQL-based constraints and targets
- **Property Path Evaluation** - Complex path expressions with performance optimization
- **Logical Constraints** - Full support for `sh:and`, `sh:or`, `sh:not`, `sh:xone`
- **Shape-based Constraints** - Nested shape validation and qualified cardinality
- **Closed Shape Validation** - Strict property closure validation

### Performance & Scalability
- **High-Performance Engine** - Optimized constraint evaluation and caching
- **Parallel Validation** - Multi-threaded validation for large datasets
- **Incremental Validation** - Delta-based validation for streaming data
- **Memory Efficient** - Minimal memory footprint for large RDF graphs
- **Index-Aware** - Leverages RDF store indexes for optimal performance

### Enterprise Features
- **Comprehensive Reporting** - Detailed violation reports with multiple output formats
- **Validation Analytics** - Performance metrics and validation statistics
- **Security Hardened** - SPARQL injection prevention and query sandboxing
- **Streaming Support** - Real-time validation for RDF streams
- **Multi-format Support** - Turtle, JSON-LD, RDF/XML, N-Triples

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
oxirs-shacl = "0.1"
oxirs-core = "0.1"
```

### Basic Usage

```rust
use oxirs_shacl::{Validator, ValidationConfig};
use oxirs_core::store::Store;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new RDF store and SHACL validator
    let store = Store::new()?;
    let mut validator = Validator::new();
    
    // Load SHACL shapes from RDF
    let shapes_ttl = r#"
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        
        ex:PersonShape a sh:NodeShape ;
            sh:targetClass ex:Person ;
            sh:property [
                sh:path ex:name ;
                sh:datatype xsd:string ;
                sh:minCount 1 ;
                sh:maxCount 1 ;
            ] ;
            sh:property [
                sh:path ex:age ;
                sh:datatype xsd:integer ;
                sh:minInclusive 0 ;
                sh:maxInclusive 150 ;
            ] .
    "#;
    
    validator.load_shapes_from_rdf(shapes_ttl, "turtle", None)?;
    
    // Load and validate data
    let data_ttl = r#"
        @prefix ex: <http://example.org/> .
        
        ex:john a ex:Person ;
            ex:name "John Doe" ;
            ex:age 30 .
            
        ex:jane a ex:Person ;
            ex:name "Jane Smith" ;
            ex:age 200 .  # Invalid: exceeds maximum age
    "#;
    
    // Parse data into store
    store.load_from_reader(data_ttl.as_bytes(), "turtle", None)?;
    
    // Validate the data
    let report = validator.validate_store(&store, None)?;
    
    // Check results
    println!("Validation conforms: {}", report.conforms());
    println!("Number of violations: {}", report.results().len());
    
    for result in report.results() {
        println!("Violation: {} at {}", 
                 result.message().unwrap_or("No message"),
                 result.focus_node().map(|n| n.to_string()).unwrap_or_default());
    }
    
    Ok(())
}
```

### Advanced Configuration

```rust
use oxirs_shacl::{ValidatorBuilder, ValidationConfig};

// Create a validator with custom configuration
let validator = ValidatorBuilder::new()
    .max_violations(100)
    .include_warnings(true)
    .fail_fast(false)
    .parallel(true)
    .timeout_ms(Some(30000))
    .max_recursion_depth(25)
    .build();
```

### Property Path Validation

```rust
// Complex property path shapes
let shapes_ttl = r#"
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix ex: <http://example.org/> .
    
    ex:PersonShape a sh:NodeShape ;
        sh:targetClass ex:Person ;
        sh:property [
            sh:path ( ex:address ex:country ) ;  # Sequence path
            sh:hasValue ex:USA ;
        ] ;
        sh:property [
            sh:path [ sh:alternativePath ( ex:email ex:phone ) ] ;  # Alternative path
            sh:minCount 1 ;
        ] .
"#;
```

### SHACL-SPARQL Constraints

```rust
// Custom SPARQL-based constraints
let shapes_ttl = r#"
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix ex: <http://example.org/> .
    
    ex:PersonShape a sh:NodeShape ;
        sh:targetClass ex:Person ;
        sh:sparql [
            a sh:SPARQLConstraint ;
            sh:message "Person's age must be consistent with birth year" ;
            sh:select """
                SELECT $this WHERE {
                    $this ex:age ?age ;
                          ex:birthYear ?birthYear .
                    BIND (YEAR(NOW()) - ?birthYear AS ?calculatedAge)
                    FILTER (ABS(?age - ?calculatedAge) > 1)
                }
            """ ;
        ] .
"#;
```

## 📊 Performance

OxiRS SHACL is designed for high-performance validation of large RDF datasets:

| Dataset Size | Validation Time | Memory Usage |
|--------------|----------------|--------------|
| 1K triples   | < 1ms          | 2MB          |
| 100K triples | < 100ms        | 25MB         |
| 1M triples   | < 1s           | 150MB        |
| 10M triples  | < 15s          | 800MB        |

*Benchmarks run on modern hardware with typical SHACL shapes*

## 🔧 Configuration Options

### Validation Configuration

```rust
use oxirs_shacl::ValidationConfig;

let config = ValidationConfig {
    max_violations: 1000,        // Limit number of violations reported
    include_info: true,          // Include info-level violations
    include_warnings: true,      // Include warning-level violations
    fail_fast: false,           // Stop on first violation
    max_recursion_depth: 50,    // Limit shape recursion depth
    timeout_ms: Some(60000),    // Validation timeout in milliseconds
    parallel: true,             // Enable parallel validation
    context: HashMap::new(),    // Custom validation context
};
```

### Feature Flags

```toml
[dependencies]
oxirs-shacl = { version = "0.1", features = ["sparql", "parallel", "async"] }
```

- **`core`** (default) - Basic SHACL Core functionality
- **`sparql`** - SHACL-SPARQL extensions
- **`parallel`** - Multi-threaded validation using Rayon
- **`async`** - Async validation support with Tokio

## 📋 Supported SHACL Features

### ✅ SHACL Core Constraints

| Constraint Component | Status | Notes |
|---------------------|--------|-------|
| `sh:class` | ✅ | Class-based validation |
| `sh:datatype` | ✅ | Datatype validation |
| `sh:nodeKind` | ✅ | Node kind constraints |
| `sh:minCount`/`sh:maxCount` | ✅ | Cardinality constraints |
| `sh:minInclusive`/`sh:maxInclusive` | ✅ | Numeric range constraints |
| `sh:minExclusive`/`sh:maxExclusive` | ✅ | Exclusive numeric ranges |
| `sh:minLength`/`sh:maxLength` | ✅ | String length constraints |
| `sh:pattern` | ✅ | Regular expression patterns |
| `sh:languageIn` | ✅ | Language tag validation |
| `sh:uniqueLang` | ✅ | Unique language constraint |
| `sh:equals` | ✅ | Value equality constraints |
| `sh:disjoint` | ✅ | Value disjointness |
| `sh:lessThan`/`sh:lessThanOrEquals` | ✅ | Comparative constraints |
| `sh:in` | ✅ | Enumeration constraints |
| `sh:hasValue` | ✅ | Required value constraints |
| `sh:closed` | ✅ | Closed shape validation |

### ✅ Logical Constraints

| Constraint | Status | Notes |
|-----------|--------|-------|
| `sh:not` | ✅ | Negation constraints |
| `sh:and` | ✅ | Conjunction constraints |
| `sh:or` | ✅ | Disjunction constraints |
| `sh:xone` | ✅ | Exclusive disjunction |

### ✅ Property Paths

| Path Type | Status | Notes |
|-----------|--------|-------|
| Sequence paths | ✅ | `( ex:prop1 ex:prop2 )` |
| Alternative paths | ✅ | `[ sh:alternativePath ( ex:prop1 ex:prop2 ) ]` |
| Inverse paths | ✅ | `[ sh:inversePath ex:prop ]` |
| Zero-or-more paths | ✅ | `[ sh:zeroOrMorePath ex:prop ]` |
| One-or-more paths | ✅ | `[ sh:oneOrMorePath ex:prop ]` |
| Zero-or-one paths | ✅ | `[ sh:zeroOrOnePath ex:prop ]` |

### ✅ Target Types

| Target Type | Status | Notes |
|-------------|--------|-------|
| `sh:targetClass` | ✅ | Class-based targeting |
| `sh:targetNode` | ✅ | Explicit node targeting |
| `sh:targetObjectsOf` | ✅ | Objects of property |
| `sh:targetSubjectsOf` | ✅ | Subjects of property |
| Implicit class targets | ✅ | Shape as class |
| SPARQL-based targets | ✅ | Custom target queries |

### ✅ SHACL-SPARQL Extensions

| Feature | Status | Notes |
|---------|--------|-------|
| `sh:sparql` constraints | ✅ | Custom SPARQL constraints |
| SPARQL-based targets | ✅ | `sh:target` with SELECT queries |
| Pre-bound variables | ✅ | `$this`, `$value`, `$PATH` |
| Custom constraint components | ✅ | Reusable SPARQL components |

## 🔍 Validation Reports

OxiRS SHACL generates comprehensive validation reports compliant with the W3C SHACL specification:

```rust
use oxirs_shacl::report::ReportFormat;

// Generate report in different formats
let report = validator.validate_store(&store, None)?;

// Turtle format (default)
let turtle_report = report.to_turtle()?;

// JSON-LD format
let jsonld_report = report.to_json_ld()?;

// JSON format (non-RDF)
let json_report = report.to_json()?;

// HTML format for web display
let html_report = report.to_html()?;
```

### Report Structure

Each validation result includes:
- **Focus Node** - The node being validated
- **Result Path** - The property path where validation failed
- **Value** - The specific value that caused the violation
- **Source Shape** - The shape that was violated
- **Source Constraint Component** - The specific constraint that failed
- **Severity** - Violation, Warning, or Info
- **Message** - Human-readable error description

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
cargo test

# Run with performance tests
cargo test --release

# Run specific test modules
cargo test constraints
cargo test validation
cargo test report

# Run W3C SHACL test suite compliance
cargo test w3c_compliance
```

## 🔧 Development

### Building from Source

```bash
git clone https://github.com/cool-japan/oxirs.git
cd oxirs/engine/oxirs-shacl
cargo build --release
```

### Running Benchmarks

```bash
cargo bench
```

### Code Quality

```bash
# Lint code
cargo clippy

# Format code
cargo fmt

# Check documentation
cargo doc --no-deps --open
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

### Development Setup

1. Install Rust (latest stable)
2. Clone the repository
3. Run `cargo test` to verify setup
4. Make your changes
5. Run `cargo clippy` and `cargo fmt`
6. Submit a pull request

## 📜 License

This project is licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🙏 Acknowledgments

- [W3C SHACL Working Group](https://www.w3.org/TR/shacl/) for the SHACL specification
- [Apache Jena](https://jena.apache.org/) for reference implementation insights
- The Rust RDF community for foundational libraries

## 📚 Related Projects

- **[oxirs-core](../oxirs-core)** - Core RDF data structures and operations
- **[oxirs-arq](../oxirs-arq)** - SPARQL query engine
- **[oxirs-fuseki](../oxirs-fuseki)** - SPARQL server with SHACL validation endpoints
- **[oxirs-shacl-ai](../../ai/oxirs-shacl-ai)** - AI-powered shape learning and optimization

---

*Part of the [OxiRS](https://github.com/cool-japan/oxirs) ecosystem - High-performance RDF tools for Rust*