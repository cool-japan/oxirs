# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OxiRS is a Rust-native, modular platform for Semantic Web, SPARQL 1.2, GraphQL, and AI-augmented reasoning. It aims to be a JVM-free alternative to Apache Jena + Fuseki with enhanced capabilities.

## Key Commands

### Build & Test
```bash
# Full build and test cycle (preferred)
./scripts/build.sh

# Build with all features
./scripts/build.sh --all-features

# Run tests continuously until no warnings (following no warnings policy)
cargo nextest run --no-fail-fast

# Linting
cargo clippy --workspace --all-targets -- -D warnings

# Format check
cargo fmt --all -- --check
```

### Development
```bash
# Setup development environment (first time)
./scripts/setup-dev.sh

# Run specific module tests
cargo nextest run -p oxirs-core --no-fail-fast

# Run the Fuseki server
cargo run -p oxirs-fuseki -- --config oxirs.toml

# CLI tool
cargo run -p oxirs -- <command>
```

## Architecture

### Module Organization
The project uses a Cargo workspace with modules organized by concern:

- **core/**: Foundation modules (oxirs-core provides RDF/SPARQL basics)
- **server/**: HTTP servers (oxirs-fuseki for SPARQL, oxirs-gql for GraphQL)
- **engine/**: Query processors (oxirs-arq, oxirs-rule, oxirs-shacl)
- **storage/**: Persistence layers (oxirs-tdb for disk storage, oxirs-cluster for distributed)
- **stream/**: Real-time processing (oxirs-stream for Kafka/NATS, oxirs-federate for federation)
- **ai/**: ML integration (oxirs-embed for embeddings, oxirs-chat for RAG)
- **tools/**: CLI utilities (oxirs command-line tool)

### Key Design Principles
1. **Incremental Adoption**: Each crate works standalone
2. **Protocol Choice**: SPARQL and GraphQL from same dataset
3. **OxiGraph Independence**: Currently extracting and adapting OxiGraph code to eliminate external dependencies
4. **Single Binary Target**: Aiming for <50MB footprint with full Jena/Fuseki feature parity

### Configuration
Uses TOML configuration (`oxirs.toml`) with sections for:
- Server settings (host, port, admin UI)
- Dataset configurations
- Security/authentication
- Feature flags (text search, vector search, RDF-star, clustering, streaming)

## Current Development Focus

Based on TODO.md, the project is actively:
1. Extracting OxiGraph code into oxirs-core to remove external dependencies
2. Implementing missing Jena/Fuseki features for compatibility
3. Building distributed storage with Raft consensus
4. Adding AI capabilities (embeddings, chat, shape learning)

## Testing Strategy

- Use `cargo nextest` exclusively (not `cargo test`)
- Run with `--nff` to see all failures (Only to check compilation errors, add --no-run)
- Each module has its own test suite
- Integration tests in `tests/` directories
- Example datasets in `data/` for testing

## Critical Dependencies

OxiRS **MUST** use SciRS2 as its scientific computing foundation (see SCIRS2_INTEGRATION_POLICY.md):
- `scirs2-core` - Core scientific primitives (REQUIRED) - **replaces direct rand and ndarray usage**
- `scirs2-graph` - Graph algorithms for RDF/knowledge graphs (when needed)
- `scirs2-linalg` - Linear algebra for embeddings/vector operations (when needed)
- `scirs2-stats` - Statistical analysis for probabilistic reasoning (when needed)
- Additional SciRS2 crates added based on compilation evidence only

SciRS2 is located at `../scirs/` relative to this project.

### FULL USE OF SciRS2-Core

OxiRS must make **FULL USE** of scirs2-core's extensive capabilities:

#### Core Array Operations (replaces ndarray)
```rust
// Use scirs2-core's ndarray extensions for vector embeddings
use scirs2_core::ndarray_ext::{Array, ArrayView, ArrayViewMut, Axis, Ix1, Ix2, IxDyn};
use scirs2_core::ndarray_ext::manipulation;  // for tensor operations in AI modules
use scirs2_core::ndarray_ext::stats;         // for similarity metrics, statistics
use scirs2_core::ndarray_ext::matrix;        // for graph adjacency matrices

// IMPORTANT: For array! macro in tests, use:
use scirs2_autograd::ndarray::array;  // array! macro is here, not in scirs2_core
```

#### Random Number Generation (replaces rand)
```rust
use scirs2_core::random::{Random, rng, DistributionExt};
use scirs2_core::random::{QuasiMonteCarloSequence, SecureRandom};  // for API keys, security
use scirs2_core::random::{ImportanceSampling, VarianceReduction};  // for probabilistic queries
```

#### Performance Optimization for Large RDF Graphs
```rust
// SIMD acceleration for graph operations
use scirs2_core::simd::{SimdArray, SimdOps, auto_vectorize};
use scirs2_core::simd_ops::{simd_dot_product, simd_matrix_multiply};

// Parallel processing for SPARQL queries
use scirs2_core::parallel::{ParallelExecutor, ChunkStrategy, LoadBalancer};
use scirs2_core::parallel_ops::{par_chunks, par_join, par_scope};

// GPU acceleration for embeddings and vector search
use scirs2_core::gpu::{GpuContext, GpuBuffer, GpuKernel, CudaBackend, MetalBackend};
use scirs2_core::tensor_cores::{TensorCore, MixedPrecision, AutoTuning};
```

#### Memory Management for Large Knowledge Graphs
```rust
// Memory-efficient RDF storage
use scirs2_core::memory_efficient::{MemoryMappedArray, LazyArray, ChunkedArray};
use scirs2_core::memory_efficient::{ZeroCopyOps, AdaptiveChunking, DiskBackedArray};

// Triple store memory management
use scirs2_core::memory::{BufferPool, GlobalBufferPool, ChunkProcessor};
use scirs2_core::memory::{LeakDetector, MemoryMetricsCollector};
```

#### Advanced Scientific Computing for AI Reasoning
```rust
// Complex numbers for Fourier transforms in signal analysis
use scirs2_core::types::{ComplexOps, ComplexExt, NumericConversion};

// Scientific constants for physical reasoning
use scirs2_core::constants::{math, physical, prefixes};
use scirs2_core::units::{UnitSystem, UnitRegistry, Dimension, convert};

// Validation for RDF data integrity
use scirs2_core::validation::{check_finite, check_in_bounds, ValidationSchema};
use scirs2_core::error::{CoreError, Result};
```

#### Production-Ready Features for SPARQL Endpoints
```rust
// Performance profiling for query optimization
use scirs2_core::profiling::{Profiler, profiling_memory_tracker};
use scirs2_core::benchmarking::{BenchmarkSuite, BenchmarkRunner};

// Metrics for SPARQL endpoint monitoring
use scirs2_core::metrics::{MetricRegistry, Counter, Gauge, Histogram, Timer};
use scirs2_core::observability::{audit, tracing};

// API versioning for backward compatibility
use scirs2_core::stability::{StabilityLevel, ApiContract, BreakingChange};
use scirs2_core::versioning::{Version, VersionManager, CompatibilityLevel};
```

#### Machine Learning Integration for Semantic AI
```rust
// ML pipeline for knowledge graph embeddings
use scirs2_core::ml_pipeline::{MLPipeline, ModelPredictor, FeatureTransformer};
use scirs2_core::ml_pipeline::{DataBatch, PipelineNode, PipelineMetrics};

// Neural architecture search for optimal embeddings
use scirs2_core::neural_architecture_search::{NeuralArchitectureSearch, SearchSpace};

// Quantum optimization for graph algorithms
use scirs2_core::quantum_optimization::{QuantumOptimizer, QuantumStrategy};
```

#### Cloud & Distributed RDF Processing
```rust
// Cloud storage for distributed triple stores
use scirs2_core::cloud::{CloudStorageClient, CloudProvider, S3, GCS, Azure};

// Distributed SPARQL query processing
use scirs2_core::distributed::{ClusterManager, JobScheduler, DataParallelism};
use scirs2_core::advanced_distributed_computing::{DistributedOptimizer, AllReduce};
```

### Mandatory Usage Guidelines

1. **NEVER** import `ndarray` directly - use `scirs2_core::ndarray_ext`
2. **NEVER** import `rand` directly - use `scirs2_core::random`
3. **ALWAYS** use scirs2-core's SIMD operations for graph algorithms
4. **ALWAYS** use scirs2-core's GPU abstractions for vector embeddings
5. **ALWAYS** use scirs2-core's memory management for large RDF datasets
6. **ALWAYS** use scirs2-core's profiling for SPARQL query optimization
7. **ALWAYS** use scirs2-core's error types and result handling
8. **EXCEPTION**: Use `scirs2_autograd::ndarray::array` for the array! macro (not available in scirs2_core)

## OxiRS Module-Specific SciRS2 Usage

### oxirs-core (RDF/SPARQL Foundation)
- Use `scirs2_core::ndarray_ext` for graph adjacency matrices
- Use `scirs2_core::simd_ops` for triple pattern matching
- Use `scirs2_core::parallel_ops` for parallel query execution
- Use `scirs2_core::memory::BufferPool` for triple caching
- Use `scirs2_graph` for graph algorithms (PageRank, centrality)

### oxirs-embed (Vector Embeddings)
- Use `scirs2_core::gpu` for GPU-accelerated embeddings
- Use `scirs2_linalg` for vector operations and similarity
- Use `scirs2_neural` for transformer-based embeddings
- Use `scirs2_core::tensor_cores` for mixed-precision training
- Use `scirs2_core::memory_efficient::LazyArray` for embedding cache

### oxirs-chat (AI-Powered Chat)
- Use `scirs2_text` for NLP preprocessing
- Use `scirs2_neural` for language models
- Use `scirs2_stats` for response ranking
- Use `scirs2_core::ml_pipeline` for RAG pipelines

### oxirs-shacl / oxirs-shacl-ai (Shape Validation)
- Use `scirs2_graph` for constraint graph analysis
- Use `scirs2_stats` for statistical shape learning
- Use `scirs2_core::validation` for constraint checking
- Use `scirs2_neural` for AI-powered shape inference

### oxirs-cluster (Distributed Storage)
- Use `scirs2_cluster` for node clustering
- Use `scirs2_core::distributed` for consensus algorithms
- Use `scirs2_optimize` for load balancing
- Use `scirs2_core::cloud` for cloud storage backends

### oxirs-stream (Real-time Processing)
- Use `scirs2_signal` for stream filtering
- Use `scirs2_stats` for stream analytics
- Use `scirs2_core::parallel_ops` for parallel stream processing
- Use `scirs2_core::memory_efficient::ChunkedArray` for buffering

### oxirs-arq (Query Engine)
- Use `scirs2_optimize` for query optimization
- Use `scirs2_stats` for query statistics
- Use `scirs2_core::profiling` for query profiling
- Use `scirs2_core::jit` for query compilation

## Development Guidelines

1. **Variable Naming**: Always use `snake_case` for variables, functions, and methods
2. **Type Naming**: Use `PascalCase` for structs, enums, traits
3. **Constants**: Use `SCREAMING_SNAKE_CASE`
4. **Workspace Dependencies**: Use `workspace = true` in Cargo.toml
5. **Latest Crates**: Always use the latest version available on crates.io
6. **Use SciRS2**: Replace direct `rand` and `ndarray` usage with `scirs2-core` equivalents
7. **File Size Limit**: Refactor files exceeding 2000 lines
8. **No Warnings**: Code must compile without any warnings
9. **Module Independence**: Each crate should be usable standalone

## Refactoring Large Files with SplitRS

For files exceeding 2000 lines or with large impl blocks (>500 lines), use [SplitRS](https://crates.io/crates/splitrs) - our production-ready AST-based refactoring tool:

### Installation

```bash
cargo install splitrs
```

### Usage

```bash
# Basic refactoring
splitrs --input src/large_file.rs --output src/large_file/

# Recommended: Split large impl blocks
splitrs \
  --input src/large_file.rs \
  --output src/large_file/ \
  --split-impl-blocks \
  --max-impl-lines 200
```

### When to Use SplitRS

| File Characteristics | Action | Tool |
|---------------------|--------|------|
| File <2000 lines | Leave as-is | - |
| Impl block <500 lines | Manual refactoring | - |
| Impl block 500-1000 lines | Consider tool | SplitRS (optional) |
| Impl block >1000 lines | **Use tool** | **SplitRS (required)** |

### Example: Refactoring OxiRS Modules

```bash
# Example: Refactor a large stream module
cd ***/oxirs
splitrs \
  --input stream/oxirs-stream/src/connection_pool.rs \
  --output stream/oxirs-stream/src/connection_pool/ \
  --split-impl-blocks \
  --max-impl-lines 200

# Review generated modules
ls stream/oxirs-stream/src/connection_pool/

# Verify compilation
cargo build -p oxirs-stream

# Run tests
cargo test -p oxirs-stream
```

### What SplitRS Does

- ✅ **AST-based analysis**: Uses `syn` for accurate Rust parsing
- ✅ **Intelligent clustering**: Groups related methods using call graphs
- ✅ **Auto-generates imports**: Context-aware `use` statements
- ✅ **Infers visibility**: Automatically applies `pub(super)`, `pub(crate)`, or `pub`
- ✅ **Handles complexity**: Generics, async, Arc/Mutex, nested types
- ✅ **Fast**: Processes 1600+ line files in <1 second

### Integration Workflow

1. **Run SplitRS**: Generate modules in a temporary directory
2. **Review output**: Check module organization and imports
3. **Verify compilation**: Ensure generated code compiles
4. **Copy to project**: Replace original file with generated modules
5. **Update parent mod.rs**: Add module declaration
6. **Test**: Run full test suite

### Example Output Structure

Input: `connection_pool.rs` (1660 lines)

Output:
```
connection_pool/
├── mod.rs                          # Module organization
├── connectionpool_type.rs          # Type definition
├── connectionpool_new_group.rs     # Constructors
├── connectionpool_acquire_group.rs # Acquisition methods
├── connectionpool_release_group.rs # Release methods
└── ... (20 more focused modules)
```

### Tool Status

- **Current version**: 0.2.0 (80% production-ready)
- **Repository**: https://github.com/cool-japan/splitrs
- **Documentation**: https://docs.rs/splitrs

SplitRS was developed during the OxiRS refactoring project and successfully refactored 32,398 lines across 17 large files.

## Common Workflows

### Importing Core Types - FULL SciRS2 Usage
```rust
// Arrays and numerical operations for embeddings
use scirs2_core::ndarray_ext::{Array, Array1, Array2, ArrayView, Ix1, Ix2, IxDyn};
use scirs2_core::ndarray_ext::stats::{mean, variance, correlation};
use scirs2_core::ndarray_ext::matrix::{eye, diag, kron};

// For array! macro in tests (SPECIAL CASE)
use scirs2_autograd::ndarray::array;

// Random number generation for API keys, sampling
use scirs2_core::random::{Random, rng, DistributionExt};

// Performance features for large RDF graphs
use scirs2_core::simd::SimdArray;
use scirs2_core::parallel_ops::{par_chunks, par_join};
use scirs2_core::gpu::{GpuContext, GpuBuffer};

// Memory efficiency for triple stores
use scirs2_core::memory_efficient::{MemoryMappedArray, LazyArray};
use scirs2_core::memory::{BufferPool, GlobalBufferPool};

// Error handling
use scirs2_core::error::{CoreError, Result};

// Profiling and metrics for SPARQL endpoints
use scirs2_core::profiling::Profiler;
use scirs2_core::metrics::{Counter, Timer};
```

### Adding RDF Processing with Full SciRS2 Integration
```rust
// oxirs-core/src/rdf/processor.rs
use scirs2_core::ndarray_ext::{Array2, ArrayView2};
use scirs2_core::random::Random;
use scirs2_core::simd_ops::simd_dot_product;
use scirs2_core::parallel_ops::par_chunks;
use scirs2_core::memory::BufferPool;
use scirs2_core::metrics::Timer;
use scirs2_core::error::Result;

pub struct RdfProcessor {
    buffer_pool: BufferPool,
    timer: Timer,
    rng: Random,
}

impl RdfProcessor {
    pub fn process_triples(&mut self, triples: ArrayView2<f32>) -> Result<Array2<f32>> {
        // Use scirs2-core's SIMD operations for similarity
        let similarity = simd_dot_product(&triples, &triples)?;

        // Use parallel processing for large graphs
        par_chunks(&triples, |chunk| {
            // Process in parallel
        });

        // Use memory-efficient operations
        let buffer = self.buffer_pool.acquire(triples.len())?;

        // Track metrics
        self.timer.record("triple_processing");

        Ok(processed)
    }
}
```

### GPU-Accelerated Embeddings with SciRS2
```rust
use scirs2_core::gpu::{GpuContext, GpuBuffer, GpuKernel};
use scirs2_core::tensor_cores::{TensorCore, MixedPrecision};

async fn compute_embeddings() -> Result<()> {
    // Use scirs2-core's GPU abstractions
    let context = GpuContext::new()?;
    let buffer = GpuBuffer::from_slice(&context, &rdf_data)?;

    // Use tensor cores for maximum performance
    let tensor_core = TensorCore::new(&context)?;
    tensor_core.gemm_mixed_precision(&nodes, &edges, &mut embeddings)?;

    Ok(())
}
```

### Benchmarking SPARQL Queries with SciRS2
```rust
use scirs2_core::benchmarking::{BenchmarkSuite, BenchmarkRunner};
use scirs2_core::profiling::Profiler;
use scirs2_core::metrics::MetricRegistry;

fn benchmark_sparql() -> Result<()> {
    let mut suite = BenchmarkSuite::new("sparql_bench");
    let profiler = Profiler::new();
    let metrics = MetricRegistry::global();

    suite.add_benchmark("select_query", |b| {
        profiler.start("sparql_select");
        // Benchmark SPARQL SELECT
        profiler.stop("sparql_select");
    });

    let results = suite.run()?;
    metrics.record_benchmark(results);

    Ok(())
}
```

### Memory-Efficient Large RDF Dataset Processing
```rust
use scirs2_core::memory_efficient::{MemoryMappedArray, AdaptiveChunking};
use scirs2_core::memory::LeakDetector;

fn process_large_rdf(path: &str) -> Result<()> {
    // Use memory-mapped arrays for huge RDF datasets
    let mmap = MemoryMappedArray::open(path)?;

    // Use adaptive chunking for optimal performance
    let chunking = AdaptiveChunking::new()
        .with_memory_limit(1 << 30)  // 1GB
        .build()?;

    // Monitor for memory leaks
    let leak_detector = LeakDetector::new();

    for chunk in mmap.chunks_adaptive(&chunking) {
        // Process RDF chunk without loading entire dataset
    }

    leak_detector.check()?;
    Ok(())
}
```

## Migration Checklist - Ensure Full SciRS2 Usage

When reviewing or writing OxiRS code, verify:

### ✅ Arrays and Numerical Operations
- [ ] NO direct `use ndarray::{...}`
- [ ] NO direct `Array`, `Array1`, `Array2` from ndarray
- [ ] YES `use scirs2_core::ndarray_ext::{Array, Array1, Array2, ...}`
- [ ] YES use scirs2-core's stats, matrix, manipulation modules
- [ ] For array! macro: `use scirs2_autograd::ndarray::array`

### ✅ Random Number Generation
- [ ] NO direct `use rand::{...}`
- [ ] NO direct `use rand_distr::{...}`
- [ ] YES `use scirs2_core::random::{Random, rng, ...}`
- [ ] YES use scirs2-core's distribution extensions

### ✅ Performance Optimization
- [ ] YES use `scirs2_core::simd` for vectorized graph operations
- [ ] YES use `scirs2_core::parallel_ops` for parallel SPARQL processing
- [ ] YES use `scirs2_core::gpu` for embedding acceleration
- [ ] YES use `scirs2_core::memory_efficient` for large RDF datasets

### ✅ Production Features
- [ ] YES use `scirs2_core::error::{CoreError, Result}`
- [ ] YES use `scirs2_core::profiling` for query performance analysis
- [ ] YES use `scirs2_core::metrics` for endpoint monitoring
- [ ] YES use `scirs2_core::benchmarking` for performance testing

### ✅ Advanced Features
- [ ] YES use `scirs2_core::ml_pipeline` for AI reasoning pipelines
- [ ] YES use `scirs2_core::jit` for SPARQL query compilation
- [ ] YES use `scirs2_core::cloud` for distributed triple storage
- [ ] YES use `scirs2_core::distributed` for federated queries

### Common Anti-Patterns to Avoid
```rust
// ❌ WRONG - Direct dependencies
use ndarray::{Array2, arr2};
use rand::Rng;
use rand_distr::Normal;

// ✅ CORRECT - Full SciRS2 usage
use scirs2_core::ndarray_ext::{Array2, arr2};
use scirs2_core::random::{Random, rng};
use scirs2_core::random::distributions::Normal;

// ✅ SPECIAL CASE - array! macro only
use scirs2_autograd::ndarray::array;  // For tests needing array! macro
```

## Important Notes

- **Jena Compatibility**: When implementing features, check Apache Jena codebase at ~/work/jena/ for reference
- **GraphQL Implementation**: Use Juniper patterns from ~/work/juniper/ as reference
- **Oxigraph Reference**: Check ~/work/oxigraph/ for RDF/SPARQL implementation patterns
- **SciRS2 Policy**: See SCIRS2_INTEGRATION_POLICY.md for detailed integration requirements
- **array! Macro**: Use `scirs2_autograd::ndarray::array` (not available in scirs2_core)

## Important Files

- `README.md` - Main project documentation
- `SCIRS2_INTEGRATION_POLICY.md` - Critical SciRS2 dependency policy
- `TODO.md` - Current development tasks
- Module-specific `TODO.md` files - Module tasks

**Remember**: OxiRS is built on the SciRS2 foundation. It must leverage the full power of the SciRS2 ecosystem to provide advanced semantic web capabilities with AI augmentation.