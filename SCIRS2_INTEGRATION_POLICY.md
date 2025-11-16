# SciRS2 Integration Policy for OxiRS

## 🚨 CRITICAL ARCHITECTURAL REQUIREMENT

**OxiRS MUST use SciRS2 as its scientific computing foundation.** This document establishes the policy for proper, minimal, and effective integration of SciRS2 crates into OxiRS.

## Core Integration Principles

### 1. **Complete SciRS2 Foundation**
- OxiRS **MUST** use SciRS2 as its complete scientific computing foundation
- **NEVER** use `rand` or `ndarray` directly - always use scirs2-core equivalents
- **REPLACE ALL** direct rand/ndarray imports with scirs2-core imports

### 2. **Mandatory Replacement Policy**
```rust
// ❌ FORBIDDEN - Direct dependencies
use rand::{thread_rng, Rng};
use ndarray::{Array2, Array3};

// ✅ REQUIRED - SciRS2 foundation
use scirs2_core::random::{Random, rng};
use scirs2_core::ndarray_ext::{Array2, Array3, array};  // array! macro included
```

### 3. **Architectural Hierarchy**
```
OxiRS (Semantic Web + SPARQL + GraphQL + AI Reasoning)
    ↓ MUST use
SciRS2-Core (Unified Scientific Computing Foundation)
    ↓ provides unified access to
ndarray, rand, num-traits, etc. (Core Rust Scientific Stack)
```

### 4. **Enforcement Policy**
- **All new code MUST use scirs2-core equivalents**
- **All existing code MUST be migrated to scirs2-core**
- **CI/CD MUST reject direct rand/ndarray imports**
- **Code reviews MUST enforce SciRS2 usage**

## Required SciRS2 Crates Analysis

### **ESSENTIAL (Always Required)**

#### `scirs2-core` - FOUNDATION
- **Use Cases**: Core scientific primitives, random number generation, numerical arrays, error handling, array! macro
- **OxiRS Modules**: ALL - replaces direct `rand` and `ndarray` usage
- **Status**: ✅ REQUIRED - Foundation crate for all numerical computations
- **Features**: `["random"]` - enabled for random number generation
- **Important**: Includes `array!` macro for creating test arrays - NO need for scirs2-autograd

### **CURRENTLY INTEGRATED**

#### `scirs2-linalg` - LINEAR ALGEBRA
- **Use Cases**: Vector operations for embeddings, similarity computations, matrix operations
- **OxiRS Modules**: `oxirs-embed/`, `oxirs-vec/`, `oxirs-chat/`
- **Status**: ✅ INTEGRATED - Used for embedding operations and vector search

#### `scirs2-stats` - STATISTICAL ANALYSIS
- **Use Cases**: Statistical inference, uncertainty quantification, probabilistic reasoning
- **OxiRS Modules**: `oxirs-rule/`, `oxirs-chat/`, `oxirs-shacl-ai/`
- **Status**: ✅ INTEGRATED - For AI reasoning and statistical analysis

#### `scirs2-cluster` - CLUSTERING
- **Use Cases**: Graph clustering, semantic grouping, distributed storage optimization
- **OxiRS Modules**: `oxirs-cluster/`, distributed consensus algorithms
- **Status**: ✅ INTEGRATED - For distributed processing and clustering

#### `scirs2-graph` - GRAPH ALGORITHMS
- **Use Cases**: RDF graph analysis, centrality metrics, path finding algorithms
- **OxiRS Modules**: `oxirs-core/`, `oxirs-rule/`, graph reasoning
- **Status**: ✅ INTEGRATED - For advanced graph algorithms on knowledge graphs

#### `scirs2-neural` - NEURAL NETWORKS
- **Use Cases**: Neural embeddings, transformer models, deep learning for reasoning
- **OxiRS Modules**: `oxirs-embed/`, `oxirs-chat/`, `oxirs-shacl-ai/`
- **Status**: ✅ INTEGRATED - For AI and neural network functionality

#### `scirs2-text` - TEXT PROCESSING
- **Use Cases**: NLP preprocessing, text search, tokenization for SPARQL queries
- **OxiRS Modules**: `oxirs-core/` (text search), `oxirs-chat/`
- **Status**: ✅ INTEGRATED - For advanced text processing and NLP

#### `scirs2-metrics` - PERFORMANCE MONITORING
- **Use Cases**: Query performance metrics, benchmark suites, system monitoring
- **OxiRS Modules**: Performance analysis, query optimization
- **Status**: ✅ INTEGRATED - For comprehensive performance monitoring

#### `scirs2-signal` - SIGNAL PROCESSING
- **Use Cases**: Stream processing, signal analysis for real-time data
- **OxiRS Modules**: `oxirs-stream/`, real-time processing
- **Status**: ✅ INTEGRATED - For streaming data analysis

#### `scirs2-optimize` - OPTIMIZATION
- **Use Cases**: Query optimization, parameter tuning, algorithm optimization
- **OxiRS Modules**: `oxirs-arq/`, query engine optimization
- **Status**: ✅ INTEGRATED - For advanced optimization algorithms

#### `scirs2-vision` - COMPUTER VISION
- **Use Cases**: Multi-modal AI, image processing for semantic data
- **OxiRS Modules**: `oxirs-chat/` (multi-modal), AI modules
- **Status**: ✅ INTEGRATED - For multi-modal AI capabilities

#### `scirs2-fft` - FAST FOURIER TRANSFORM
- **Use Cases**: Signal analysis, frequency domain processing
- **OxiRS Modules**: `oxirs-stream/`, signal processing components
- **Status**: ✅ INTEGRATED - For advanced signal processing

### **FUTURE CONSIDERATIONS**

#### `scirs2-sparse` - SPARSE MATRICES
- **Use Cases**: Sparse adjacency matrices for large RDF graphs, memory-efficient representations
- **OxiRS Modules**: `oxirs-core/`, `oxirs-tdb/` (future optimization)
- **Status**: 🔶 FUTURE - For memory-optimized large graph storage

#### `scirs2-datasets` - DATA HANDLING
- **Use Cases**: Benchmark RDF datasets, synthetic knowledge graph generation
- **OxiRS Modules**: Testing, benchmarking infrastructure
- **Status**: 🔶 FUTURE - For enhanced benchmark and test data

#### `scirs2-transform` - MATHEMATICAL TRANSFORMS
- **Use Cases**: Advanced mathematical transforms for semantic reasoning
- **OxiRS Modules**: Advanced reasoning components (future)
- **Status**: 🔶 FUTURE - For specialized mathematical operations

#### `scirs2-interpolate` - INTERPOLATION
- **Use Cases**: Data interpolation for sparse knowledge graphs
- **OxiRS Modules**: AI reasoning components (future)
- **Status**: 🔶 FUTURE - For advanced data interpolation

#### `scirs2-integrate` - NUMERICAL INTEGRATION
- **Use Cases**: Probabilistic reasoning, uncertainty quantification
- **OxiRS Modules**: Advanced reasoning modules (future)
- **Status**: 🔶 FUTURE - For probabilistic integration

#### `scirs2-special` - SPECIAL FUNCTIONS
- **Use Cases**: Specialized mathematical functions for advanced AI
- **OxiRS Modules**: Advanced AI reasoning (future)
- **Status**: 🔶 FUTURE - For specialized mathematical operations

#### `scirs2-io` - INPUT/OUTPUT
- **Use Cases**: Scientific data formats, if needed beyond RDF/SPARQL
- **OxiRS Modules**: Data import/export (future)
- **Status**: 🔶 FUTURE - For scientific data format support

### **NOT CURRENTLY NEEDED**

#### `scirs2-ndimage` - IMAGE PROCESSING
- **Status**: ❌ NOT NEEDED - Specific image processing beyond vision module

#### `scirs2-series` - TIME SERIES
- **Status**: ❌ NOT NEEDED - Temporal functionality handled by other modules

## Integration Guidelines

### **Adding New SciRS2 Dependencies**

1. **Document Justification**
   ```markdown
   ## SciRS2 Crate Addition Request

   **Crate**: scirs2-[name]
   **Requestor**: [Developer Name]
   **Date**: [Date]

   **Justification**:
   - Specific OxiRS feature requiring this crate
   - Code modules that will use it
   - Alternatives considered and why SciRS2 is preferred

   **Impact Assessment**:
   - Compilation time impact
   - Binary size impact
   - Maintenance burden
   ```

2. **Code Review Requirements**
   - Demonstrate actual usage in OxiRS code
   - Show integration examples
   - Verify no equivalent functionality exists in already-included crates

3. **Documentation Requirements**
   - Update this policy document
   - Document usage patterns in relevant module docs
   - Add examples to integration tests

### **Removing SciRS2 Dependencies**

1. **Regular Audits** (quarterly)
   - Review all SciRS2 dependencies for actual usage
   - Remove unused imports and dependencies
   - Update documentation

2. **Deprecation Process**
   - Mark as deprecated with removal timeline
   - Provide migration guide if functionality moves
   - Remove after deprecation period

### **Best Practices**

1. **Import Granularity**
   ```rust
   // ✅ GOOD - Specific imports
   use scirs2_core::random::Random;
   use scirs2_graph::algorithms::PageRank;

   // ❌ BAD - Broad imports
   use scirs2_core::*;
   use scirs2_graph::*;
   ```

2. **Feature Gates**
   ```rust
   // ✅ GOOD - Optional features
   #[cfg(feature = "ai-reasoning")]
   use scirs2_neural::embeddings::Embedding;
   ```

3. **Error Handling**
   ```rust
   // ✅ GOOD - Proper error context
   use scirs2_core::ScientificNumber;
   // Document why SciRS2 types are used over alternatives
   ```

### **Known Issues and Solutions**

1. **array! Macro Usage Pattern**

   **Key Point**: The `array!` macro is now available in `scirs2_core::ndarray_ext`

   **Correct Usage Pattern**:
   ```rust
   // All array operations use scirs2_core
   use scirs2_core::ndarray_ext::{Array, Array1, Array2, ArrayView, array};
   use scirs2_core::ndarray_ext::stats::{mean, variance};

   // Example usage
   let data = Array2::zeros((3, 3));      // Use scirs2_core types
   let test_arr = array![[1, 2], [3, 4]]; // Use scirs2_core array! macro
   ```

   **Important**:
   - `scirs2_core::ndarray_ext::*` - For ALL array operations including array! macro
   - ❌ FORBIDDEN: `scirs2_autograd::ndarray::array` - DO NOT USE

2. **Complete Migration Pattern from Direct Dependencies**
   ```rust
   // ========== RANDOM NUMBER GENERATION ==========
   // OLD: use rand::{thread_rng, Rng, random};
   // NEW: use scirs2_core::random::{Random, rng, DistributionExt};

   // OLD: use rand_distr::Normal;
   // NEW: use scirs2_core::random::distributions::Normal;

   // ========== ARRAY OPERATIONS ==========
   // OLD: use ndarray::{Array1, Array2, Array3, ArrayView, Axis};
   // NEW: use scirs2_core::ndarray_ext::{Array1, Array2, Array3, ArrayView, Axis};

   // OLD: use ndarray::{arr1, arr2, array};
   // NEW: use scirs2_core::ndarray_ext::{arr1, arr2, array};  // array! macro included

   // ❌ FORBIDDEN: use scirs2_autograd::ndarray::array;  // DO NOT USE

   // ========== LINEAR ALGEBRA ==========
   // OLD: Manual matrix operations with ndarray
   // NEW: use scirs2_linalg::{Matrix, Vector, LinearAlgebra};

   // ========== SIMD OPERATIONS ==========
   // OLD: Manual vectorization
   // NEW: use scirs2_core::simd::{SimdArray, SimdOps, auto_vectorize};

   // ========== ERROR HANDLING ==========
   // OLD: anyhow::Error or custom errors
   // NEW: use scirs2_core::error::{CoreError, Result}; // When appropriate
   ```

## Enforcement

### **Automated Checks**
- CI pipeline checks for unused SciRS2 dependencies
- Documentation tests verify integration examples work
- Dependency graph analysis in builds

### **Manual Reviews**
- All SciRS2 integration changes require team review
- Quarterly dependency audits
- Annual architecture review

### **Violation Response**
1. **Warning**: Document why integration is needed
2. **Correction**: Remove unjustified dependencies
3. **Training**: Educate team on integration policy

## Future Considerations

### **SciRS2 Version Management**
- Track SciRS2 release cycle
- Test OxiRS against SciRS2 beta releases
- Coordinate breaking change migrations

### **Performance Monitoring**
- Benchmark impact of SciRS2 integration
- Monitor compilation times
- Track binary size impact

### **Community Alignment**
- Coordinate with SciRS2 team on roadmap
- Contribute improvements back to SciRS2
- Maintain architectural consistency

## Conclusion

This policy ensures OxiRS properly leverages SciRS2's scientific computing foundation while maintaining a clean, minimal, and justified dependency graph. **OxiRS must use SciRS2, but intelligently and purposefully.**

---

**Document Version**: 2.0
**Last Updated**: 2025-09-22
**Next Review**: 2026-02-22 (Quarterly)
**Owner**: OxiRS Architecture Team

## Quick Reference

### Current OxiRS Integration (Production Setup)
```toml
# Essential SciRS2 dependencies for OxiRS (from workspace Cargo.toml)
scirs2 = { version = "0.1.0-beta.2" }
scirs2-core = { version = "0.1.0-beta.2", features = ["random"] }  # Foundation (includes array! macro)

# Production integrated crates
scirs2-linalg = { version = "0.1.0-beta.2" }     # Vector operations, embeddings
scirs2-stats = { version = "0.1.0-beta.2" }      # Statistical analysis, AI reasoning
scirs2-neural = { version = "0.1.0-beta.2" }     # Neural networks, deep learning
scirs2-graph = { version = "0.1.0-beta.2" }      # Graph algorithms on RDF
scirs2-metrics = { version = "0.1.0-beta.2" }    # Performance monitoring
scirs2-fft = { version = "0.1.0-beta.2" }        # Signal processing
scirs2-signal = { version = "0.1.0-beta.2" }     # Stream processing
scirs2-optimize = { version = "0.1.0-beta.2" }   # Query optimization
scirs2-cluster = { version = "0.1.0-beta.2" }    # Distributed clustering
scirs2-text = { version = "0.1.0-beta.2" }       # NLP and text processing
scirs2-vision = { version = "0.1.0-beta.2" }     # Multi-modal AI capabilities
```

**Remember**: Start minimal, add based on evidence, document everything!