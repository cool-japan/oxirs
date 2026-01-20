use thiserror::Error;

/// Errors that can occur in time-series operations
#[derive(Error, Debug)]
pub enum TsdbError {
    /// I/O error during storage operations
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Compression or decompression error
    #[error("Compression error: {0}")]
    Compression(String),

    /// Decompression error
    #[error("Decompression error: {0}")]
    Decompression(String),

    /// Invalid series ID
    #[error("Invalid series ID: {0}")]
    InvalidSeriesId(u64),

    /// Invalid time range
    #[error("Invalid time range: start={start}, end={end}")]
    InvalidTimeRange {
        /// Start timestamp (milliseconds since epoch)
        start: i64,
        /// End timestamp (milliseconds since epoch)
        end: i64,
    },

    /// Series not found
    #[error("Series not found: {0}")]
    SeriesNotFound(u64),

    /// Chunk not found
    #[error("Chunk not found for series {series_id} at {timestamp}")]
    ChunkNotFound {
        /// Series ID
        series_id: u64,
        /// Timestamp
        timestamp: i64,
    },

    /// Write buffer full
    #[error("Write buffer full: {0} points pending")]
    BufferFull(usize),

    /// WAL error
    #[error("Write-Ahead Log error: {0}")]
    Wal(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Query error
    #[error("Query error: {0}")]
    Query(String),

    /// Integration error (RDF store integration)
    #[error("Integration error: {0}")]
    Integration(String),
}

/// Result type for time-series operations
pub type TsdbResult<T> = Result<T, TsdbError>;
