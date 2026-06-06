//! # Event Serialization Module
//!
//! This module provides comprehensive serialization support for stream events with:
//! - Multiple format support (JSON, Protobuf, Avro, Binary)
//! - Schema evolution and versioning
//! - Compression integration
//! - Format auto-detection
//! - Schema registry integration
//!
//! Implementation is split across sibling modules:
//! - `serialization_types`: Format enum, schema registry, options, delta types, Protobuf/Avro
//! - `serialization_encoder`: `EventSerializer`, `FormatConverter`, `StreamingSerializer`,
//!   `EnhancedBinaryFormat`
//! - `serialization_decoder`: `DeltaCompressor` (delta compression/decompression)
//! - `serialization_tests`: unit tests for the public API

pub use crate::serialization_decoder::*;
pub use crate::serialization_encoder::*;
pub use crate::serialization_types::*;
