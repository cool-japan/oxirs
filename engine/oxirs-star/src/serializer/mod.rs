//! RDF-star serialization module
//!
//! This module provides serializers for RDF-star formats including:
//! - Turtle-star (*.ttls)
//! - N-Triples-star (*.nts)
//! - TriG-star (*.trigs)
//! - N-Quads-star (*.nqs)
//! - JSON-LD-star (*.jlds)
//!
//! Features:
//! - Streaming serialization for large datasets
//! - Compression support (gzip, zstd)
//! - Parallel serialization for multi-core systems
//! - Memory-efficient processing with buffer reuse
//! - Configurable batching and buffering strategies

pub mod config;
pub mod parallel;
pub mod simd_escape;
pub mod star_serializer;
pub mod streaming;

// Re-export main types
pub use config::{CompressionType, SerializationOptions, StreamingConfig};
pub use parallel::ParallelSerializer;
pub use simd_escape::SimdEscaper;
pub use star_serializer::StarSerializer;
pub use streaming::StreamingSerializer;
