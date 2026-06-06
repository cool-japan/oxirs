//! Serialization configuration for OxiRS Core.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializationConfig {
    pub default_format: SerializationFormat,
    pub formats: HashMap<String, FormatConfig>,
    pub output: OutputConfig,
    pub compression: CompressionConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializationFormat {
    NTriples,
    Turtle,
    RdfXml,
    JsonLd,
    NQuads,
    TriG,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatConfig {
    pub name: String,
    pub pretty_print: bool,
    pub indent_size: usize,
    pub line_length: usize,
    pub options: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub encoding: String,
    pub buffer_size: usize,
    pub enable_buffering: bool,
    pub flush_interval: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub enabled: bool,
    pub algorithm: CompressionAlgorithm,
    pub level: u8,
    pub min_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Gzip,
    Bzip2,
    Lz4,
    Zstd,
}

impl Default for SerializationConfig {
    fn default() -> Self {
        Self {
            default_format: SerializationFormat::Turtle,
            formats: HashMap::new(),
            output: OutputConfig::default(),
            compression: CompressionConfig::default(),
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            encoding: "UTF-8".to_string(),
            buffer_size: 8192,
            enable_buffering: true,
            flush_interval: Duration::from_millis(100),
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: CompressionAlgorithm::Gzip,
            level: 6,
            min_size: 1024,
        }
    }
}
