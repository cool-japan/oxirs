//! DiskANN - Disk-based Approximate Nearest Neighbor Search
//!
//! Implementation of Microsoft's DiskANN algorithm for billion-scale vector search.
//! DiskANN is optimized for SSD storage and can handle datasets too large for RAM.
//!
//! ## Key Features
//! - Disk-based storage with memory-mapped files
//! - SSD-optimized I/O patterns
//! - Vamana graph for efficient navigation
//! - Streaming search without full index loading
//! - Incremental updates support
//!
//! ## Architecture
//! - **In-memory layer**: Hot vectors and navigation graph
//! - **Disk layer**: Full vector dataset on SSD
//! - **Compressed layer**: Optionally compressed vectors (PQ)
//!
//! ## References
//! - DiskANN paper: <https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf>

pub mod builder;
pub mod config;
pub mod graph;
pub mod index;
pub mod search;
pub mod storage;
pub mod types;

pub use builder::{DiskAnnBuildStats, DiskAnnBuilder};
pub use config::{DiskAnnConfig, PruningStrategy, SearchMode};
pub use graph::{VamanaGraph, VamanaNode};
pub use index::{DiskAnnIndex, IndexMetadata};
pub use search::{BeamSearch, SearchStats};
pub use storage::{DiskStorage, MemoryMappedStorage, StorageBackend};
pub use types::{DiskAnnError, DiskAnnResult, NodeId, VectorId};
