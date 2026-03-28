# OxiRS Vec Documentation

**Version**: v0.2.3
**Status**: Production-Ready RC

Welcome to the OxiRS Vec documentation! This comprehensive guide covers everything you need to deploy, optimize, and operate OxiRS Vec in production.

## Table of Contents

1. [Deployment Guide](#deployment-guide)
2. [Performance Tuning Guide](#performance-tuning-guide)
3. [Best Practices Guide](#best-practices-guide)
4. [WAL Configuration Guide](#wal-configuration-guide)
5. [GPU Acceleration Guide](#gpu-acceleration-guide)

---

## Quick Links

### Getting Started

- **New to OxiRS Vec?** Start with the [Deployment Guide](oxirs-vec-deployment-guide.md)
- **Need to optimize performance?** See the [Performance Tuning Guide](oxirs-vec-performance-tuning-guide.md)
- **Looking for best practices?** Read the [Best Practices Guide](oxirs-vec-best-practices.md)

### Advanced Topics

- **Setting up crash recovery?** Check the [WAL Configuration Guide](oxirs-vec-wal-guide.md)
- **Want GPU acceleration?** Follow the [GPU Acceleration Guide](oxirs-vec-gpu-acceleration-guide.md)

---

## Documentation Overview

### 📘 [Deployment Guide](oxirs-vec-deployment-guide.md)

**Essential reading for production deployment**

Learn how to deploy OxiRS Vec in production environments:
- Architecture overview
- System requirements and capacity planning
- Installation and configuration
- Deployment patterns (single-node, multi-tenant, distributed)
- Monitoring and observability
- Security best practices
- Disaster recovery
- Migration from FAISS/Annoy

**Who should read**: DevOps engineers, System administrators, Production engineers

**Topics covered**:
- ✅ Production architecture
- ✅ System requirements
- ✅ Configuration examples
- ✅ Monitoring setup
- ✅ Security hardening
- ✅ Disaster recovery
- ✅ Troubleshooting

---

### 🚀 [Performance Tuning Guide](oxirs-vec-performance-tuning-guide.md)

**Optimize OxiRS Vec for your specific workload**

Master performance optimization techniques:
- Index algorithm selection (HNSW, IVF, PQ, DiskANN, LSH)
- HNSW parameter tuning
- Memory optimization (quantization, compression)
- Query optimization (caching, batching, rewriting)
- GPU acceleration
- Workload-specific tuning

**Who should read**: Performance engineers, ML engineers, Backend developers

**Topics covered**:
- ✅ Algorithm comparison and selection
- ✅ Parameter tuning matrices
- ✅ Memory reduction techniques (4-32× savings)
- ✅ Query optimization strategies
- ✅ Caching strategies
- ✅ Benchmarking tools

**Performance Goals**:
- Query Latency (p95): < 50 ms
- Throughput: > 1000 QPS
- Recall@10: > 0.95
- Memory efficiency: 4-6 GB per 1M vectors

---

### 📖 [Best Practices Guide](oxirs-vec-best-practices.md)

**Production-proven practices for robust applications**

Build reliable and efficient vector search applications:
- Data modeling (dimensionality, normalization)
- Index design (selection, building, updates)
- Query patterns (KNN, filtering, caching)
- Error handling and validation
- Testing strategies
- Monitoring and alerting
- Security (authentication, rate limiting)
- Maintenance procedures

**Who should read**: All developers using OxiRS Vec

**Topics covered**:
- ✅ Data modeling best practices
- ✅ Index design patterns
- ✅ Query optimization
- ✅ Error handling
- ✅ Testing guidelines
- ✅ Security practices
- ✅ Common anti-patterns to avoid

**Key Principles**:
- Plan for 10× scale
- Measure everything
- Test thoroughly
- Security first
- Simplicity wins

---

### 💾 [WAL Configuration Guide](oxirs-vec-wal-guide.md)

**Configure crash recovery and durability**

Set up Write-Ahead Logging for data durability:
- WAL architecture and concepts
- Configuration options
- Crash recovery procedures
- Performance tuning
- Monitoring and maintenance
- Advanced topics (2PC, PITR, replication)

**Who should read**: DBAs, DevOps engineers, System administrators

**Topics covered**:
- ✅ WAL fundamentals
- ✅ Durability vs performance trade-offs
- ✅ Automatic recovery
- ✅ Manual recovery procedures
- ✅ Performance optimization
- ✅ Troubleshooting

**Durability Options**:
- **Maximum Durability**: 0ms sync interval (20-30% overhead)
- **Balanced**: 100ms sync interval (5-10% overhead)
- **Performance-Optimized**: 5s sync interval (1-2% overhead)

---

### ⚡ [GPU Acceleration Guide](oxirs-vec-gpu-acceleration-guide.md)

**Leverage GPU for 10-50× speedup**

Accelerate vector operations with CUDA:
- System requirements (CUDA, GPU compatibility)
- Installation and setup
- Supported operations (16 distance metrics)
- Performance benchmarks
- Memory management
- Optimization techniques
- Troubleshooting

**Who should read**: ML engineers, Performance engineers, AI researchers

**Topics covered**:
- ✅ CUDA installation
- ✅ GPU configuration
- ✅ Batch processing (10-50× speedup)
- ✅ Mixed precision (FP16, 2× speedup)
- ✅ Tensor Cores (8× for matrix ops)
- ✅ Multi-GPU setup
- ✅ Benchmarking

**Performance Gains**:
| Operation | CPU | GPU (RTX 3090) | Speedup |
|-----------|-----|----------------|---------|
| Cosine | 100k/s | 5M/s | 50× |
| Euclidean | 120k/s | 4M/s | 33× |
| Dot Product | 150k/s | 8M/s | 53× |

---

## OxiRS Vec at a Glance

### Current Status (v0.2.3)

- ✅ **667 tests passing** (100% pass rate)
- ✅ **Production-ready**: Real-time updates, crash recovery, multi-tenancy
- ✅ **Advanced indexing**: HNSW, IVF, PQ/OPQ, LSH, DiskANN, Learned Indexes
- ✅ **20+ distance metrics**: Comprehensive similarity functions
- ✅ **GPU acceleration**: CUDA kernels for 16 metrics
- ✅ **Hybrid search**: Keyword + semantic with BM25
- ✅ **SPARQL integration**: Custom functions, federated queries

### Key Features

#### Indexing Algorithms
- **HNSW**: Best all-around (recall: 0.98, latency: 8ms)
- **IVF**: Fast search (recall: 0.92, latency: 5ms)
- **LSH**: Ultra-fast approximate (recall: 0.85, latency: 3ms)
- **PQ/OPQ**: Low memory (32× compression)
- **DiskANN**: Billion-scale (minimal memory)
- **NSG**: Graph-based balanced (recall: 0.96)

#### Performance
- **Query Latency**: p50 < 10ms, p95 < 50ms, p99 < 100ms
- **Throughput**: 1000+ QPS (single-node)
- **Recall**: > 0.95 for most use cases
- **Scalability**: Supports 100M+ vectors

#### Production Features
- **Multi-tenancy**: Tenant isolation, quotas, billing
- **Real-time updates**: Streaming ingestion
- **Crash recovery**: WAL-based durability
- **Monitoring**: Metrics, alerts, health checks
- **Caching**: Multi-level query result caching
- **Security**: Authentication, rate limiting, validation

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/cool-japan/oxirs.git
cd oxirs/engine/oxirs-vec

# Build
cargo build --release --features hnsw,simd,parallel,gpu

# Run tests
cargo test --release
```

### Basic Usage

```rust
use oxirs_vec::{VectorStore, embeddings::EmbeddingStrategy};

fn main() -> anyhow::Result<()> {
    // Create vector store
    let mut store = VectorStore::with_embedding_strategy(
        EmbeddingStrategy::SentenceTransformer
    )?;

    // Index documents
    store.index_resource("doc1".to_string(), "Rust programming")?;
    store.index_resource("doc2".to_string(), "Machine learning")?;

    // Search
    let results = store.similarity_search("programming languages", 10)?;

    for (uri, score) in results {
        println!("{}: {:.3}", uri, score);
    }

    Ok(())
}
```

---

## Documentation Roadmap

### Completed ✅
- [x] Production deployment guide
- [x] Performance tuning guide
- [x] Best practices guide
- [x] WAL configuration guide
- [x] GPU acceleration guide

### Planned 📋
- [ ] Migration guide from FAISS (detailed)
- [ ] Migration guide from Annoy
- [ ] API reference documentation
- [ ] Integration guides (Python, Node.js)
- [ ] Use case examples
- [ ] Advanced SPARQL integration
- [ ] Distributed deployment guide

---

## Support & Resources

### Documentation
- **API Docs**: https://docs.rs/oxirs-vec
- **GitHub**: https://github.com/cool-japan/oxirs
- **Issues**: https://github.com/cool-japan/oxirs/issues

### Community
- **Discussions**: https://github.com/cool-japan/oxirs/discussions
- **Examples**: See `examples/` directory

### Related Projects
- **OxiRS Core**: RDF/SPARQL foundation
- **OxiRS Fuseki**: SPARQL server
- **OxiRS GraphQL**: GraphQL endpoint

---

## Contributing

We welcome contributions to documentation! To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Documentation guidelines:
- Write in clear, concise English
- Include code examples
- Add diagrams where helpful
- Test all code snippets
- Follow existing formatting

---

## License

OxiRS Vec is licensed under the Apache License, Version 2.0. See [LICENSE](../LICENSE) for details.

---

## Acknowledgments

OxiRS Vec builds on research and implementations from:
- **HNSW**: Malkov & Yashunin (2016)
- **IVF**: Jégou et al. (2011)
- **Product Quantization**: Jégou et al. (2011)
- **DiskANN**: Subramanya et al. (2019)
- **Learned Indexes**: Kraska et al. (2018)

Special thanks to the Rust community and contributors.

---

**Last Updated**: 2026-01-06
**Document Version**: 1.0
**OxiRS Vec Version**: v0.2.3
