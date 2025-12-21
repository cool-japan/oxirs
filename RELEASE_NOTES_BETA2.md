# OxiRS v0.1.0-beta.2 Release Notes

**Release Date**: December 21, 2025
**Release Type**: Beta (Production-Ready)
**Status**: Released
**Stability**: 98% API Stable

---

## Announcing OxiRS v0.1.0-beta.2

We're excited to announce **OxiRS v0.1.0-beta.2**, a significant milestone advancing OxiRS toward the v1.0.0 stable release. This release focuses on **stabilizing distributed storage, AI modules, and performance optimization** while maintaining full backward compatibility with beta.1.

### What is OxiRS?

OxiRS is a **Rust-native, modular platform** for Semantic Web, SPARQL 1.2, GraphQL, and AI-augmented reasoning. It serves as a **JVM-free alternative** to Apache Jena + Fuseki with:

- **5-10x faster** query execution
- **2x more memory efficient**
- **Zero-copy operations** and vectorized execution
- **Distributed storage** with Raft consensus
- **AI capabilities** (embeddings, chat, shape learning)
- **Production-ready** monitoring and security

---

## Key Highlights

### 1. **CUDA GPU Acceleration for Embeddings**
- **Tucker decomposition** with CUDA support for knowledge graph embeddings
- **2-5x faster** embedding computation on NVIDIA GPUs
- **Multi-GPU support** for large-scale knowledge graphs
- **Metal backend** support for Apple Silicon

### 2. **Memory-Mapped Storage Optimization**
- **Mmap optimizer** for TDB storage with intelligent caching
- **Reduced memory footprint** for large datasets (>10GB)
- **Faster startup times** with lazy loading
- **Background writeback** for durability

### 3. **Enhanced Vision-Language Integration**
- **Vision-Language Graph (VLG)** support for multimodal RDF
- **Image embeddings** integrated with knowledge graphs
- **CLIP-style** semantic similarity search
- **Cross-modal queries** combining text and image

### 4. **AI Module Stabilization**
- **oxirs-embed**: Promoted from experimental to unstable (75% frozen)
- **oxirs-chat**: Promoted from experimental to unstable (70% frozen)
- **RAG pipelines**: Production-ready with session management
- **Multi-LLM support**: OpenAI, Anthropic Claude, Ollama

### 5. **Performance Improvements**

| Operation | Beta.1 | Beta.2 | Improvement |
|-----------|--------|--------|-------------|
| Simple SELECT (1M triples) | 10ms p95 | **8ms p95** | 20% faster |
| Embedding generation (GPU) | 100ms | **40ms** | 2.5x faster |
| TDB startup (10GB) | 5s | **2s** | 2.5x faster |
| Memory usage (10M triples) | 5GB | **4GB** | 20% reduction |

### 6. **Test Suite Expansion**
- **12,248 tests** across all modules (100% pass rate)
- **Comprehensive integration tests** for new features
- **Property-based tests** for edge cases
- **Regression tests** for all fixed bugs

---

## What's New in Beta.2

### Core Features

#### 1. CUDA GPU Acceleration
```rust
use oxirs_embed::models::tucker::{TuckerConfig, TuckerDecomposition};

// Enable CUDA acceleration
let config = TuckerConfig {
    embedding_dim: 256,
    device: Device::Cuda(0),  // Use first CUDA device
    ..Default::default()
};

let tucker = TuckerDecomposition::new(config)?;
let embeddings = tucker.fit_transform(&knowledge_graph)?;
```

#### 2. Memory-Mapped Storage
```rust
use oxirs_tdb::storage::MmapOptimizer;

// Configure mmap optimization
let optimizer = MmapOptimizer::builder()
    .cache_size_mb(1024)
    .lazy_loading(true)
    .background_sync(true)
    .build()?;

let store = TdbStore::with_optimizer(path, optimizer)?;
```

#### 3. Vision-Language Graph
```rust
use oxirs_embed::vision_language_graph::VisionLanguageGraph;

// Create multimodal knowledge graph
let vlg = VisionLanguageGraph::new(config)?;
vlg.add_image_entity("product_image.jpg", &product_uri)?;
vlg.link_image_to_entity(&product_uri, &product_node)?;

// Cross-modal query
let similar = vlg.find_similar_to_image("query_image.jpg", top_k)?;
```

#### 4. RAG Pipeline with Session Management
```rust
use oxirs_chat::{ChatSession, RAGEngine};

// Create persistent chat session
let session = ChatSession::builder()
    .session_id("user_123")
    .max_history(50)
    .persist_to_disk(true)
    .build()?;

let rag = RAGEngine::new(&dataset, &embedding_model)?;
let response = rag.chat(&session, "What products are similar to X?").await?;
```

### Infrastructure Improvements

#### 5. Enhanced Monitoring
- **New Prometheus metrics** for GPU utilization
- **Mmap cache hit/miss ratios**
- **Embedding generation latency histograms**
- **Session activity metrics**

#### 6. Improved Error Handling
- **Structured errors** with error codes
- **Recovery suggestions** in error messages
- **Debug context** for troubleshooting
- **Chain error support** for root cause analysis

---

## Module Stability Updates

| Module | Beta.1 | Beta.2 | Production Ready? |
|--------|--------|--------|-------------------|
| **oxirs-core** | 🟢 Stable (95%) | 🟢 Stable (96%) | Yes |
| **oxirs-arq** | 🟢 Stable (90%) | 🟢 Stable (92%) | Yes |
| **oxirs-fuseki** | 🟢 Stable (95%) | 🟢 Stable (96%) | Yes |
| **oxirs-tdb** | 🟢 Stable (90%) | 🟢 Stable (93%) | Yes |
| **oxirs-cluster** | 🟡 Unstable (70%) | 🟢 Stable (90%) | Yes |
| **oxirs-gql** | 🟡 Unstable (80%) | 🟢 Stable (90%) | Yes |
| **oxirs-shacl** | 🟡 Unstable (75%) | 🟡 Unstable (80%) | With caution |
| **oxirs-embed** | 🔴 Experimental (60%) | 🟡 Unstable (75%) | With caution |
| **oxirs-chat** | 🔴 Experimental (50%) | 🟡 Unstable (70%) | With caution |
| **oxirs-stream** | 🟡 Unstable (65%) | 🟡 Unstable (75%) | With caution |
| **oxirs-shacl-ai** | 🔴 Experimental (50%) | 🔴 Experimental (55%) | Research only |

**Legend**:
- 🟢 **Stable**: Safe for production, backward compatible
- 🟡 **Unstable**: Use with caution, may change in v0.2.0
- 🔴 **Experimental**: Research preview, no stability guarantees

---

## Migration from Beta.1

### Summary

- **100% backward compatible** - All Beta.1 code works without changes
- **No data migration required** - Same TDB format
- **No deprecated APIs removed** - All Beta.1 deprecations still work
- **Configuration additive** - New sections are optional
- **Estimated migration time**: Immediate (no changes required)

### New Configuration Options (Optional)

```toml
# oxirs.toml (new sections in Beta.2)

[storage.mmap]
enabled = true
cache_size_mb = 1024
lazy_loading = true

[ai.cuda]
enabled = true
device_id = 0
memory_limit_gb = 8

[chat.session]
persist = true
max_history = 50
expiration_hours = 24
```

---

## Project Statistics

- **Total Lines of Code**: ~1.6M lines
- **Rust Code**: ~1.3M lines
- **Documentation**: ~62K lines in markdown
- **Test Count**: 12,248 tests (100% pass rate)
- **Crates**: 22 modules in workspace
- **Contributors**: Growing community

---

## Known Issues

### Critical (Blocking Production)
- None

### Major (Workarounds Available)
- **Issue #234**: CUDA memory fragmentation on long-running embedding jobs
  - **Workaround**: Restart embedding service every 24 hours
  - **Fix planned**: v0.1.1 (patch release)

### Minor (Low Impact)
- **Issue #256**: Mmap cache metrics may show stale values for up to 30 seconds
  - **Impact**: Dashboard shows slightly delayed cache statistics
  - **Fix planned**: v0.2.0

**Full issue tracker**: https://github.com/cool-japan/oxirs/issues

---

## Roadmap to v1.0.0

### v0.1.0-beta.2 (Current - December 2025)
- CUDA GPU acceleration for embeddings
- Memory-mapped storage optimization
- AI module stabilization (embed, chat)
- Vision-language graph support

### v0.1.0-rc.1 (Q1 2026)
- Full API freeze (99% stable)
- External security audit
- Performance validation
- Documentation completion

### v1.0.0 (Q2 2026)
- Long-term API stability guarantee (10 years)
- SOC 2 / ISO 27001 certification
- LTS release with 3-year support

---

## Documentation

### New in Beta.2
- [CUDA Acceleration Guide](docs/CUDA_GUIDE.md)
- [Memory-Mapped Storage Guide](docs/MMAP_GUIDE.md)
- [Vision-Language Graph Tutorial](docs/VLG_TUTORIAL.md)
- [Chat Session Management](docs/CHAT_SESSIONS.md)

### Getting Started
- **Quick Start**: [QUICKSTART.md](tools/oxirs/QUICKSTART.md)
- **Command Reference**: [COMMAND_REFERENCE.md](tools/oxirs/docs/COMMAND_REFERENCE.md)
- **Configuration**: [CONFIGURATION.md](tools/oxirs/docs/CONFIGURATION.md)

### Production Deployment
- **Docker Deployment**: [docs/DEPLOYMENT.md#docker-deployment](docs/DEPLOYMENT.md#docker-deployment)
- **Kubernetes Deployment**: [docs/DEPLOYMENT.md#kubernetes-deployment](docs/DEPLOYMENT.md#kubernetes-deployment)
- **Monitoring Setup**: [docs/DEPLOYMENT.md#monitoring-and-logging](docs/DEPLOYMENT.md#monitoring-and-logging)

### API Reference
- **docs.rs**: https://docs.rs/oxirs-core/0.1.0-beta.2
- **GitHub**: https://github.com/cool-japan/oxirs

---

## Downloads

### Pre-built Binaries
- **Linux (x86_64)**: [oxirs-linux-x86_64.tar.gz](https://github.com/cool-japan/oxirs/releases/download/v0.1.0-beta.2/oxirs-linux-x86_64.tar.gz)
- **Linux (x86_64, CUDA)**: [oxirs-linux-x86_64-cuda.tar.gz](https://github.com/cool-japan/oxirs/releases/download/v0.1.0-beta.2/oxirs-linux-x86_64-cuda.tar.gz)
- **macOS (ARM64)**: [oxirs-macos-arm64.tar.gz](https://github.com/cool-japan/oxirs/releases/download/v0.1.0-beta.2/oxirs-macos-arm64.tar.gz)
- **Windows (x86_64)**: [oxirs-windows-x86_64.zip](https://github.com/cool-japan/oxirs/releases/download/v0.1.0-beta.2/oxirs-windows-x86_64.zip)

### Docker Images
```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/cool-japan/oxirs-fuseki:v0.1.0-beta.2

# With CUDA support
docker pull ghcr.io/cool-japan/oxirs-fuseki:v0.1.0-beta.2-cuda
```

### Source Code
```bash
# Clone from GitHub
git clone --branch v0.1.0-beta.2 https://github.com/cool-japan/oxirs.git

# Build from source
cd oxirs
cargo build --release

# Build with CUDA support
cargo build --release --features cuda
```

---

## License

OxiRS is licensed under **Apache License 2.0** or **MIT License** (dual-licensed).

- **Apache-2.0**: [LICENSE-APACHE](LICENSE-APACHE)
- **MIT**: [LICENSE-MIT](LICENSE-MIT)

---

## Summary

**OxiRS v0.1.0-beta.2** advances toward production stability with:

- **CUDA GPU acceleration** for 2-5x faster embeddings
- **Memory-mapped storage** for efficient large dataset handling
- **AI module stabilization** (embed, chat promoted to unstable)
- **Vision-language graph** for multimodal RDF processing
- **12,248 tests** with 100% pass rate
- **Full backward compatibility** with Beta.1

**Ready for**: Production deployments, AI-powered knowledge graph applications

**Target**: v1.0.0 stable release in **Q2 2026** with 10-year API stability guarantee

---

## Next Steps

1. **Try it out**: Follow the [Quick Start](tools/oxirs/QUICKSTART.md) guide
2. **Enable CUDA**: See [CUDA Guide](docs/CUDA_GUIDE.md) for GPU acceleration
3. **Explore AI features**: Try [oxirs-chat](ai/oxirs-chat/) for RAG over RDF
4. **Join the community**: [GitHub Discussions](https://github.com/cool-japan/oxirs/discussions)
5. **Report feedback**: Help us reach v1.0.0!

---

*OxiRS v0.1.0-beta.2 Release Notes*
*Release Date: December 21, 2025*
*Status: Production-Ready (98% Complete)*

**Thank you for using OxiRS!**
