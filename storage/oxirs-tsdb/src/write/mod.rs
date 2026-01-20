//! Write path with WAL and compaction
//!
//! Provides durable writes with Write-Ahead Log and in-memory buffering
//! for high-throughput data ingestion.

pub mod buffer;
pub mod compactor;
pub mod retention;
pub mod wal;

pub use buffer::{BufferConfig, BufferStats, WriteBuffer};
pub use compactor::{CompactionConfig, CompactionStats, Compactor};
pub use retention::{RetentionEnforcer, RetentionStats};
pub use wal::{WalEntry, WriteAheadLog};
