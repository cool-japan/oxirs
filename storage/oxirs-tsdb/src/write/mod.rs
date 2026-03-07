//! Write path with WAL and compaction
//!
//! Provides durable writes with Write-Ahead Log and in-memory buffering
//! for high-throughput data ingestion.
//!
//! ## Modules
//!
//! - `buffer` -- Async write buffer with series-level flushing
//! - `compactor` -- Background compaction of in-memory chunks to disk
//! - `retention` -- Time-based data expiry and downsampling
//! - `wal` -- Original binary WAL format (24 bytes/record, no CRC)
//! - `batch_writer` -- High-performance batched writer with CRC32-protected WAL

pub mod batch_writer;
pub mod buffer;
pub mod compactor;
pub mod retention;
pub mod wal;

pub use batch_writer::{BatchWriter, BatchWriterConfig, CrcWal, MetricPoint};
pub use buffer::{BufferConfig, BufferStats, WriteBuffer};
pub use compactor::{CompactionConfig, CompactionStats, Compactor};
pub use retention::{RetentionEnforcer, RetentionStats};
pub use wal::{WalEntry, WriteAheadLog};
