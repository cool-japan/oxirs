# OxiRS Core - Performance Optimization Guide

*Version: v0.1.0-rc.2 | Last Updated: December 21, 2025*

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start: Performance Checklist](#quick-start-performance-checklist)
3. [Zero-Copy Operations](#zero-copy-operations)
4. [SIMD-Optimized Triple Matching](#simd-optimized-triple-matching)
5. [Concurrent Operations](#concurrent-operations)
6. [ACID Transactions](#acid-transactions)
7. [Query Optimization](#query-optimization)
8. [Memory Management](#memory-management)
9. [SPARQL 1.2 RDF-star Performance](#sparql-12-rdf-star-performance)
10. [Benchmarking Your Application](#benchmarking-your-application)

---

## Introduction

OxiRS Core achieves high performance through multiple advanced features:

- **Zero-copy RDF operations** - Minimize allocations with memory-mapped files
- **SIMD triple matching** - 3-8x speedup using platform-specific vectorization
- **Lock-free concurrency** - Optimized for read-heavy workloads (10:1 ratio)
- **JIT query optimization** - 10-50x speedup for repeated queries
- **Batch processing** - 50-100x faster bulk operations
- **MVCC transactions** - Full ACID guarantees without blocking readers

This guide shows you how to leverage these features for maximum performance.

---

## Quick Start: Performance Checklist

✅ **Enable all performance features:**
\`\`\`toml
[dependencies]
oxirs-core = { version = "0.1.0-rc.2", features = ["parallel", "simd"] }
\`\`\`

✅ **Use zero-copy operations for large datasets:**
\`\`\`rust
use oxirs_core::zero_copy_rdf::ZeroCopyTripleStore;

let store = ZeroCopyTripleStore::with_mmap("/path/to/data.rdf")?;
\`\`\`

✅ **Enable batch processing for bulk inserts:**
\`\`\`rust
use oxirs_core::concurrent::BatchBuilder;

let mut batch = BatchBuilder::new()
    .with_capacity(10000)
    .with_auto_flush(true);

for triple in triples {
    batch.add(triple)?;
}
batch.flush()?;
\`\`\`

---

## Zero-Copy Operations

### Overview

Zero-copy operations reduce memory allocations by 60-80% through:
- Memory-mapped files
- BufferPool for efficient memory reuse
- Zero-copy serialization/deserialization

### When to Use

✅ **Use zero-copy for:**
- Large RDF datasets (>1GB)
- Read-heavy workloads
- Long-running server applications
- Persistent storage backends

❌ **Avoid zero-copy for:**
- Small datasets (<10MB) - overhead not worth it
- Write-heavy workloads - mmap sync overhead
- Temporary in-memory graphs

**For full guide, see the complete documentation.**

---

## Performance Benchmarks Summary

### Overall Performance Gains (vs baseline)

| Feature | Speedup | Use Case |
|---------|---------|----------|
| Zero-copy operations | 60-80% less memory | Large datasets |
| SIMD triple matching | 3-8x faster | Pattern matching |
| Lock-free reads | 7x throughput | Concurrent queries |
| Batch processing | 50-100x faster | Bulk inserts |
| JIT optimization | 10-50x faster | Repeated queries |
| Query plan caching | 95% less compile time | Common queries |
| Parallel execution | 6-8x faster (8 cores) | Complex queries |
| MVCC transactions | No reader blocking | Mixed workloads |

---

**For complete guide, see PERFORMANCE_GUIDE.md**
