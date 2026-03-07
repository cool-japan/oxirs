//! Advanced compression algorithms for time-series data
//!
//! This module provides a suite of compression algorithms optimised for
//! different time-series data patterns:
//!
//! | Module       | Algorithm  | Best suited for |
//! |-------------|------------|----------------|
//! | `gorilla`  | Gorilla XOR | Floating-point sensor values |
//! | `rle`      | RLE        | Step-function / constant-value data |
//! | `dictionary` | Dictionary | Repeated categorical strings |
//! | `adaptive` | Adaptive   | Unknown/mixed data patterns |

pub mod adaptive;
pub mod dictionary;
pub mod gorilla;
pub mod rle;

// Re-exports for convenience
pub use adaptive::{AdaptiveCompressor, CompressedBlock, CompressionAlgorithm, SampleStats};
pub use dictionary::{dict_decode, dict_encode, DictionaryBlock, DictionaryEncoder};
pub use gorilla::{gorilla_decode, gorilla_encode, GorillaDecoder, GorillaEncoder, GORILLA_MAGIC};
pub use rle::{rle_decode, rle_encode, RleBlock, RleEncoder, RleRun};
