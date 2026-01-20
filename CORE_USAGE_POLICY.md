# Core Usage Policy for OxiRS

This document outlines the mandatory policies for using oxirs-core modules across the entire OxiRS ecosystem. All contributors and modules MUST adhere to these guidelines to ensure consistency, maintainability, and optimal performance.

## Table of Contents

1. [Overview](#overview)
2. [SIMD Operations Policy](#simd-operations-policy)
3. [GPU Operations Policy](#gpu-operations-policy)
4. [Parallel Processing Policy](#parallel-processing-policy)
5. [BLAS Operations Policy](#blas-operations-policy)
6. [Platform Detection Policy](#platform-detection-policy)
7. [Performance Optimization Policy](#performance-optimization-policy)
8. [Error Handling Policy](#error-handling-policy)
9. [Memory Management Policy](#memory-management-policy)
10. [Refactoring Guidelines](#refactoring-guidelines)
11. [Examples](#examples)

## Overview

The oxirs-core crate serves as the central hub for all common functionality, optimizations, and abstractions used across OxiRS modules. This centralized approach ensures:

- **Consistency**: All modules use the same optimized implementations
- **Maintainability**: Updates and improvements are made in one place
- **Performance**: Optimizations are available to all modules
- **Portability**: Platform-specific code is isolated in core

## SIMD Operations Policy

### Mandatory Rules

1. **ALWAYS use `oxirs-core::simd` module** for all SIMD operations
2. **NEVER implement custom SIMD** code in individual modules
3. **NEVER use direct SIMD libraries** (wide, packed_simd, std::arch) in modules
4. **ALWAYS provide scalar fallbacks** through the unified trait

### Required Usage Pattern

```rust
use oxirs_core::simd::{SimdOps, SimdVector};

// CORRECT - Uses unified SIMD operations
let result = SimdOps::add(&a, &b);
let dot_product = SimdOps::dot(&x, &y);

// INCORRECT - Direct SIMD implementation
// use std::arch::x86_64::*;  // FORBIDDEN in modules
// unsafe { _mm256_add_ps(...) }  // FORBIDDEN
```

### Available SIMD Operations

All operations are available through the `oxirs-core::simd` module:

- `add`, `sub`, `mul`, `div` - Element-wise operations
- `dot` - Dot product
- `cosine_distance` - Cosine distance calculation
- `euclidean_distance` - Euclidean distance calculation
- `manhattan_distance` - Manhattan distance calculation
- `norm` - L2 norm
- `max`, `min` - Element-wise min/max
- `sum`, `mean` - Reductions
- `fma` - Fused multiply-add

## GPU Operations Policy

### Mandatory Rules

1. **ALWAYS use `oxirs-core::gpu` module** for all GPU operations
2. **NEVER implement direct CUDA/OpenCL/Metal kernels** in modules
3. **NEVER make direct GPU API calls** outside of core
4. **ALWAYS register GPU kernels** in the core GPU kernel registry

### GPU Backend Support

The core GPU module provides unified abstractions for:
- CUDA
- ROCm
- WebGPU
- Metal
- OpenCL

### Usage Pattern

```rust
use oxirs_core::gpu::{GpuDevice, GpuKernel};

// CORRECT - Uses core GPU abstractions
let device = GpuDevice::default()?;
let kernel = device.compile_kernel(KERNEL_SOURCE)?;

// INCORRECT - Direct CUDA usage
// use cuda_sys::*;  // FORBIDDEN in modules
```

## Parallel Processing Policy

### Mandatory Rules

1. **ALWAYS use `oxirs-core::parallel` module** for all parallel operations
2. **NEVER add direct `rayon` dependency** to module Cargo.toml files
3. **ALWAYS import via `use oxirs_core::parallel::*`**
4. **NEVER use `rayon::prelude::*` directly** in modules

### Required Usage Pattern

```rust
// CORRECT - Uses core parallel abstractions
use oxirs_core::parallel::*;

let results: Vec<i32> = (0..1000)
    .into_par_iter()
    .map(|x| x * x)
    .collect();

// INCORRECT - Direct Rayon usage
// use rayon::prelude::*;  // FORBIDDEN in modules
```

### Features Provided

The `parallel` module provides:

- **Full Rayon functionality** when `parallel` feature is enabled
- **Sequential fallbacks** when `parallel` feature is disabled
- **Helper functions**:
  - `par_chunks(slice, size)` - Process slices in parallel chunks
  - `par_scope(closure)` - Execute in parallel scope
  - `par_join(a, b)` - Execute two closures in parallel
- **Runtime detection**:
  - `is_parallel_enabled()` - Check if parallel processing is available
  - `num_threads()` - Get number of threads for parallel operations

### Module Dependencies

```toml
# CORRECT - Module Cargo.toml
[dependencies]
oxirs-core = { workspace = true, features = ["parallel"] }

# INCORRECT - Direct Rayon dependency
# rayon = { workspace = true }  # FORBIDDEN
```

## BLAS Operations Policy

### Mandatory Rules

1. **ALL BLAS operations go through `oxirs-core`**
2. **NEVER add direct BLAS dependencies** to individual modules
3. **Backend selection is handled by core's platform configuration**
4. **Use feature flags through core** for BLAS backend selection

### Supported BLAS Backends

- macOS: Accelerate Framework (default)
- Linux/Windows: OpenBLAS (default)
- Intel MKL (optional)
- Netlib (fallback)

### Module Dependencies

```toml
# CORRECT - Module Cargo.toml
[dependencies]
oxirs-core = { workspace = true, features = ["blas"] }

# INCORRECT - Direct BLAS dependency
# openblas-src = "0.10"  # FORBIDDEN
```

## Platform Detection Policy

### Mandatory Rules

1. **ALWAYS use `oxirs-core::platform::PlatformCapabilities`** for capability detection
2. **NEVER implement custom CPU feature detection**
3. **NEVER duplicate platform detection code**

### Usage Pattern

```rust
use oxirs_core::platform::PlatformCapabilities;

// CORRECT - Uses core platform detection
let caps = PlatformCapabilities::detect();
if caps.simd_available {
    // Use SIMD path
}

// INCORRECT - Custom detection
// if is_x86_feature_detected!("avx2") {  // FORBIDDEN
```

### Available Capabilities

- `simd_available` - SIMD support
- `gpu_available` - GPU support
- `cuda_available` - CUDA support
- `opencl_available` - OpenCL support
- `metal_available` - Metal support (macOS)
- `avx2_available` - AVX2 instructions
- `avx512_available` - AVX512 instructions
- `neon_available` - ARM NEON instructions

## Performance Optimization Policy

### Automatic Optimization Selection

Use `oxirs-core::optimizer::AutoOptimizer` for automatic selection:

```rust
use oxirs_core::optimizer::AutoOptimizer;

let optimizer = AutoOptimizer::new();

// Automatically selects best implementation based on problem size
if optimizer.should_use_gpu(problem_size) {
    // Use GPU implementation from core
} else if optimizer.should_use_simd(problem_size) {
    // Use SIMD implementation from core
} else {
    // Use scalar implementation
}
```

### Required Core Features

Each module should enable relevant core features:

```toml
[dependencies]
oxirs-core = { workspace = true, features = ["simd", "parallel", "gpu", "blas"] }
```

## Error Handling Policy

### Mandatory Rules

1. **Base all module errors on `oxirs-core::error`**
2. **Provide proper error conversions** to/from core errors
3. **Use core validation functions** for parameter checking

### Usage Pattern

```rust
use oxirs_core::error::CoreError;
use oxirs_core::validation::{check_positive, check_finite};

// Module-specific error should derive from core
#[derive(Debug, thiserror::Error)]
pub enum ModuleError {
    #[error(transparent)]
    Core(#[from] CoreError),
    // Module-specific variants...
}

// Use core validation
check_positive(value, "parameter_name")?;
check_finite(&array)?;
```

## Memory Management Policy

### Mandatory Rules

1. **Use `oxirs-core::memory` algorithms** for large data
2. **Use `oxirs-core::cache` for caching** instead of custom solutions
3. **Follow core memory pooling strategies** when available

### Available Memory-Efficient Operations

- `chunk_wise_op` - Process large arrays in chunks
- `streaming_op` - Stream processing for very large data
- Memory pools for temporary allocations

### Caching

```rust
use oxirs_core::cache::{CacheBuilder, TTLCache};

// CORRECT - Uses core caching
let cache = CacheBuilder::new()
    .max_size(100)
    .ttl(Duration::from_secs(60))
    .build();

// INCORRECT - Custom caching
// let mut cache = HashMap::new();  // Don't implement custom caching
```

## Refactoring Guidelines

When encountering code that violates these policies, follow this priority order:

1. **SIMD implementations** - Replace all custom SIMD with `oxirs-core::simd`
2. **GPU implementations** - Centralize all GPU kernels in `oxirs-core::gpu`
3. **Parallel operations** - Replace direct Rayon usage with `oxirs-core::parallel`
4. **Platform detection** - Replace with `PlatformCapabilities::detect()`
5. **BLAS operations** - Ensure all go through core
6. **Caching mechanisms** - Replace custom caching with core implementations
7. **Error types** - Base on core error types
8. **Validation** - Use core validation functions

## Examples

### Example 1: Vector Operations

```rust
use oxirs_core::simd::SimdOps;

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Use unified SIMD operations - no direct SIMD code
    1.0 - SimdOps::cosine_distance(a, b)
}
```

### Example 2: Adaptive Implementation

```rust
use oxirs_core::optimizer::AutoOptimizer;
use oxirs_core::simd::SimdOps;

pub fn process_vectors(data: &[f32]) -> f32 {
    let optimizer = AutoOptimizer::new();
    let size = data.len();
    
    if optimizer.should_use_simd(size) {
        // Automatically uses SIMD if available
        SimdOps::sum(data) / size as f32
    } else {
        // Falls back to scalar
        data.iter().sum::<f32>() / size as f32
    }
}
```

### Example 3: Platform-Aware Code

```rust
use oxirs_core::platform::PlatformCapabilities;

pub fn get_optimization_info() -> String {
    let caps = PlatformCapabilities::detect();
    
    format!(
        "Available optimizations: {}",
        caps.summary()
    )
}
```

### Example 4: Parallel Processing

```rust
use oxirs_core::parallel::*;

pub fn parallel_distance_matrix(points: &[f32]) -> Vec<f32> {
    // Works with or without parallel feature
    let distances: Vec<f32> = (0..points.len())
        .into_par_iter()
        .map(|i| {
            // Complex computation for each point
            compute_distance(points[i])
        })
        .collect();
    
    distances
}

pub fn adaptive_processing(data: &[f32]) -> f32 {
    if is_parallel_enabled() && data.len() > 1000 {
        // Use parallel processing for large datasets
        data.into_par_iter()
            .map(|&x| x * x)
            .sum::<f32>()
    } else {
        // Use sequential for small datasets
        data.iter()
            .map(|&x| x * x)
            .sum()
    }
}
```

## Enforcement

- Code reviews MUST check for policy compliance
- CI/CD pipelines should include linting for direct SIMD/GPU usage
- Regular audits should identify and refactor non-compliant code
- New modules MUST follow these policies from the start

## Benefits

By following these policies, we achieve:

1. **Unified Performance**: All modules benefit from optimizations
2. **Easier Maintenance**: Updates in one place benefit all modules
3. **Consistent Behavior**: Same optimizations across the ecosystem
4. **Better Testing**: Centralized testing of critical operations
5. **Improved Portability**: Platform-specific code is isolated
6. **Reduced Duplication**: No repeated implementation of common operations

## Questions or Clarifications

If you have questions about these policies or need clarification on specific use cases, please:

1. Check the `oxirs-core` documentation
2. Review existing implementations in other modules
3. Open an issue for discussion
4. Consult with the core team

Remember: When in doubt, use the core abstractions!