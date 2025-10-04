//! # OxiRS TDB - Apache Jena TDB/TDB2 Compatible Storage Engine
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0--alpha.2-orange)](https://github.com/cool-japan/oxirs/releases)
//!
//! **Status**: Alpha Release (v0.1.0-alpha.2)
//! ⚠️ APIs may change. Not recommended for production use.
//!
//! High-performance RDF triple store with B+Tree indexes, ACID transactions,
//! and Apache Jena TDB/TDB2 API compatibility.
//!
//! ## Features
//!
//! - **B+Tree Storage** - Efficient range queries and sequential scans
//! - **Triple Indexes** - SPO, POS, OSP for optimal query performance
//! - **ACID Transactions** - Write-Ahead Logging (WAL) with 2PL concurrency control
//! - **Dictionary Encoding** - Compress IRIs and literals to 8-byte NodeIDs
//! - **Buffer Pool** - LRU caching for hot pages
//! - **100M+ Triples** - Scalable to large datasets
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use oxirs_tdb::TDBStore;
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a new TDB store
//! let store = TDBStore::open("/path/to/data")?;
//!
//! // Insert triples
//! // store.insert(...)?;
//!
//! // Query triples
//! // let results = store.query(...)?;
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │         TDBStore (High-level API)       │
//! └─────────────────┬───────────────────────┘
//!                   │
//!       ┌───────────┴───────────┐
//!       │                       │
//! ┌─────▼─────┐         ┌──────▼──────┐
//! │  Indexes  │         │ Dictionary  │
//! │ SPO/POS/  │         │   Encoding  │
//! │    OSP    │         │             │
//! └─────┬─────┘         └──────┬──────┘
//!       │                      │
//!       └───────────┬──────────┘
//!                   │
//!            ┌──────▼──────┐
//!            │   B+Tree    │
//!            └──────┬──────┘
//!                   │
//!       ┌───────────┴───────────┐
//!       │                       │
//! ┌─────▼─────┐         ┌──────▼──────┐
//! │  Buffer   │         │     WAL     │
//! │   Pool    │         │  (Logging)  │
//! └─────┬─────┘         └──────┬──────┘
//!       │                      │
//!       └───────────┬──────────┘
//!                   │
//!            ┌──────▼──────┐
//!            │ File Manager│
//!            │   (mmap)    │
//!            └─────────────┘
//! ```
//!
//! ## See Also
//!
//! - [`oxirs-core`](https://docs.rs/oxirs-core) - RDF data model
//! - [`oxirs-arq`](https://docs.rs/oxirs-arq) - SPARQL query engine

#![doc(html_root_url = "https://docs.rs/oxirs-tdb/0.1.0-alpha.2")]
#![warn(missing_docs)]
#![allow(dead_code)] // Allow during development
#![allow(unused_imports)] // Allow during development
#![allow(unused_variables)] // Allow during development

// Core modules
pub mod error;

// Storage layer
pub mod storage;

// B+Tree implementation
pub mod btree;

// Index structures
pub mod index;

// Dictionary encoding
pub mod dictionary;

// Transaction management
pub mod transaction;

// Compression and optimization
pub mod compression;

// High-level store API
pub mod store;

// Re-export commonly used types
pub use error::{Result, TdbError};
pub use store::{TdbConfig, TdbStats, TdbStore};

/// TDB storage engine version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default page size (4KB)
pub const DEFAULT_PAGE_SIZE: usize = 4096;

/// Default buffer pool size (1000 pages = 4MB)
pub const DEFAULT_BUFFER_POOL_SIZE: usize = 1000;
